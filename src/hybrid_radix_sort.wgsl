
struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    node_index: u32,
    material_id: u32,
    flags: u32,
};


struct PrefixUniforms {
    // 0-7
    pass_number: u32,
};

const NUM_BANKS = 32u;
const LOG_NUM_BANKS = 5u;
const BITS_PER_KEY = 64u;
const BITS_PER_DIGIT = 8u;
const KEYS_PER_THREAD = 9u;
// 9 * 256
const KEYS_PER_BLOCK = 2304u;
//delta hat
const LOCAL_SORTING_THRESHOLD = 4224u;
//delta underscore
const MERGING_BUCKET_THRESHOLD = 3000u;

// assignment of thread blocks to key blocks
// corresponds to "Block assignments"
// starting offset of the keys, number of consecutive keys, buckets unique identifier, bucket offset
struct ThreadBlockKeyBlockMap {
    starting_key_offset: u32,
    consecutive_key_count: u32,
    bucket_id: u32,
    bucket_offset: u32,
};

// assignment of a bucket whose size falls short of the local sort threshold
// corresponds to "Local sort sub-bucket assignments"
struct SmallBucket {
    bucket_id: u32,
    bucket_offset: u32,
    is_merged: bool
};

struct BlockHistogram {
    vals: array<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: PrefixUniforms;

@group(0) @binding(1)
var<storage, read> triangles: array<TriangleIndices>;

@group(0) @binding(2)
var<storage, read> codes: array<u32>;

@group(0) @binding(3)
var<storage, read> triangles_2: array<TriangleIndices>;

@group(0) @binding(4)
var<storage, read> codes_2: array<u32>;

@group(0) @binding(5)
var<storage, read_write> block_histograms: array<BlockHistogram>;

var<workgroup, read_write> histogram: array<atomic<u32>, 256>;
var<workgroup, read_write> histogram_prefix_sum: array<u32, 256>;

// var<workgroup, read_write> shared_mem: array<u32, MERGING_BUCKET_THRESHOLD>;

// Returns the offset needed to avoid a bank offset
fn bank_offset(index: u32) -> u32 {
    return index >> LOG_NUM_BANKS;
}

fn get_digit_from_pass_number(id: u32) -> u32 {
    if uniforms.pass_number < 4u {
        //grab from hi bits
        //24, 16, 8 , 0
        let bit_shift: u32 = ((3u - uniforms.pass_number) * 8u);
        return (codes[id * 2u] >> bit_shift) & 0xFFu;
    } else {
        //grab from lo bits
        let bit_shift: u32 = ((3u - (uniforms.pass_number % 4u)) * 8u);
        return (codes[id * 2u + 1u] >> bit_shift) & 0xFFu;
    }
}

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = histogram_prefix_sum[wid + lo];
    let after = histogram_prefix_sum[wid + hi];
    histogram_prefix_sum[wid + lo] = after;
    histogram_prefix_sum[wid + hi] += before;
}

@compute @workgroup_size(256, 1, 1)
fn hybrid_radix_sort(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;
    let v = codes[get_digit_from_pass_number(id)];

    if uniforms.pass_number == 0u {
        // always do counting sort first
        let digit = codes[get_digit_from_pass_number(id * 9u)];

        // compute histogram
        for (var i: u32 = 0u; i < KEYS_PER_THREAD; i++) {
            atomicAdd(&histogram[digit], 1u);
        }

        // exclusive prefix-sum
        workgroupBarrier();
        // move the memory to non-atomic storage
        histogram_prefix_sum[wid] = atomicLoad(&histogram[wid]);
        workgroupBarrier();
        // up-sweep
        if wid % 2u == 0u {
            // 0, 2, 4, 6, 8, ...
            // blue arrows: 0 -> 1, 2 -> 3
            // black arrows: 1 -> 1, 3 -> 3
            histogram_prefix_sum[wid + 1u] += histogram_prefix_sum[wid];
        }
        workgroupBarrier();
        if wid % 4u == 0u {
            // 0, 4, 8, 12, 16, ...
            // blue arrows: 1 -> 3, 5 -> 7
            // black arrows: 3 -> 3, 7 -> 7
            histogram_prefix_sum[wid + 3u] += histogram_prefix_sum[wid + 1u];
        }
        workgroupBarrier();
        if wid % 8u == 0u {
            // 0, 8, 16, 24, 32, ...
            // blue arrows: 3 -> 7
            // black arrows: 7 -> 7
            histogram_prefix_sum[wid + 7u] += histogram_prefix_sum[wid + 3u];
        }
        workgroupBarrier();
        if wid % 16u == 0u {
            // 0, 16, 32, 48, 64, ...
            histogram_prefix_sum[wid + 15u] += histogram_prefix_sum[wid + 7u];
        }
        workgroupBarrier();
        if wid % 32u == 0u {
            // 0, 32, 64, ...
            histogram_prefix_sum[wid + 31u] += histogram_prefix_sum[wid + 15u];
        }
        workgroupBarrier();
        if wid % 64u == 0u {
            // 0, 64, 128, 192, ...
            histogram_prefix_sum[wid + 63u] += histogram_prefix_sum[wid + 31u];
        }
        workgroupBarrier();
        if wid % 128u == 0u {
            // 0, 128, ...
            histogram_prefix_sum[wid + 127u] += histogram_prefix_sum[wid + 63u];
        }
        workgroupBarrier();
        if wid % 256u == 0u {
            // 0
            histogram_prefix_sum[wid + 255u] += histogram_prefix_sum[wid + 127u];
        }
        workgroupBarrier();
        // down-sweep
        if wid % 256u == 0u {
            let before = histogram_prefix_sum[wid + 127u];
            histogram_prefix_sum[wid + 127u] = 0u;
            histogram_prefix_sum[wid + 255u] = before;
        }
        workgroupBarrier();
        if wid % 128u == 0u {
            prefix_sum_swap(wid, 63u, 127u);
        }
        workgroupBarrier();
        if wid % 64u == 0u {
            prefix_sum_swap(wid, 31u, 63u);
        }
        workgroupBarrier();
        if wid % 32u == 0u {
            prefix_sum_swap(wid, 15u, 31u);
        }
        workgroupBarrier();
        if wid % 16u == 0u {
            prefix_sum_swap(wid, 7u, 15u);
        }
        workgroupBarrier();
        if wid % 8u == 0u {
            prefix_sum_swap(wid, 3u, 7u);
        }
        workgroupBarrier();
        if wid % 4u == 0u {
            prefix_sum_swap(wid, 1u, 3u);
        }
        workgroupBarrier();
        if wid % 2u == 0u {
            prefix_sum_swap(wid, 0u, 1u);
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < KEYS_PER_THREAD; i++) {
            atomicAdd(&histogram[digit], 1u);
        }
    } else {
    }
}
