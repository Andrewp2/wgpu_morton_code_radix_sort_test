struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    // node_index: u32,
    // material_id: u32,
    // flags: u32,
};

struct PrefixUniforms {
    pass_number: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: PrefixUniforms;

@group(0) @binding(1)
var<storage, read> triangles: array<TriangleIndices>;

@group(0) @binding(2)
var<storage, read> codes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> triangles_2: array<TriangleIndices>;

@group(0) @binding(4)
var<storage, read_write> codes_2: array<u32>;

@group(0) @binding(5)
var<storage, read> storage_histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 256>;
var<workgroup> histogram_na: array<u32, 256>;
var<workgroup> offset_bmp: array<u32, 2048>; // 8 u32 values for each of 256 threads
var<workgroup> local_offset: array<u32, 256>;
var<workgroup> ballot_mem: array<atomic<u32>, 8>;

fn select_digit(code_1: u32, code_2: u32) -> u32 {
    var val = 256u;
    if uniforms.pass_number < 4u {
        val = (code_1 >> (uniforms.pass_number * 8u)) & 255u;
    } else {
        val = (code_2 >> ((uniforms.pass_number - 4u) * 8u)) & 255u;
    }
    return val;
}

fn prefix_sum_block_exclusive_unrolled(wid: u32) {
    if wid % 2u == 0u {
        histogram_na[wid + 1u] += histogram_na[wid];
    }
    if wid % 4u == 0u {
        histogram_na[wid + 3u] += histogram_na[wid + 1u];
    }
    if wid % 8u == 0u {
        histogram_na[wid + 7u] += histogram_na[wid + 3u];
    }
    if wid % 16u == 0u {
        histogram_na[wid + 15u] += histogram_na[wid + 7u];
    }
    if wid % 32u == 0u {
        histogram_na[wid + 31u] += histogram_na[wid + 15u];
    }
    workgroupBarrier();
    if wid % 64u == 0u {
        histogram_na[wid + 63u] += histogram_na[wid + 31u];
    }
    workgroupBarrier();
    if wid % 128u == 0u {
        histogram_na[wid + 127u] += histogram_na[wid + 63u];
    }
    workgroupBarrier();
    if wid % 256u == 0u {
        histogram_na[wid + 255u] += histogram_na[wid + 127u];
    }
    workgroupBarrier();
    // special case for first iteration
    if wid % 256u == 0u {
        let before = histogram_na[127u];
        histogram_na[127u] = 0u;
        histogram_na[255u] = before;
    }
    var i = 128u;
    workgroupBarrier();
    if wid % 128u == 0u {
        let lo = 63u;
        let hi = 127u;
        let before = histogram_na[wid + lo];
        let after = histogram_na[wid + hi];
        histogram_na[wid + lo] = after;
        histogram_na[wid + hi] += before;
    }
    workgroupBarrier();
    if wid % 64u == 0u {
        let lo = 31u;
        let hi = 63u;
        let before = histogram_na[wid + lo];
        let after = histogram_na[wid + hi];
        histogram_na[wid + lo] = after;
        histogram_na[wid + hi] += before;
    }
    workgroupBarrier();
    if wid % 32u == 0u {
        let lo = 15u;
        let hi = 31u;
        let before = histogram_na[wid + lo];
        let after = histogram_na[wid + hi];
        histogram_na[wid + lo] = after;
        histogram_na[wid + hi] += before;
    }
    if wid % 16u == 0u {
        let lo = 7u;
        let hi = 15u;
        let before = histogram_na[wid + lo];
        let after = histogram_na[wid + hi];
        histogram_na[wid + lo] = after;
        histogram_na[wid + hi] += before;
    }
    if wid % 8u == 0u {
        let lo = 3u;
        let hi = 7u;
        let before = histogram_na[wid + lo];
        let after = histogram_na[wid + hi];
        histogram_na[wid + lo] = after;
        histogram_na[wid + hi] += before;
    }
    if wid % 4u == 0u {
        let lo = 1u;
        let hi = 3u;
        let before = histogram_na[wid + lo];
        let after = histogram_na[wid + hi];
        histogram_na[wid + lo] = after;
        histogram_na[wid + hi] += before;
    }
    if wid % 2u == 0u {
        let hi = 1u;
        let before = histogram_na[wid];
        let after = histogram_na[wid + hi];
        histogram_na[wid] = after;
        histogram_na[wid + hi] += before;
    }
    workgroupBarrier();
}

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = histogram_na[wid + lo];
    let after = histogram_na[wid + hi];
    histogram_na[wid + lo] = after;
    histogram_na[wid + hi] += before;
}

fn prefix_sum_block_exclusive(wid: u32) {
    for (var i: u32 = 1u; i < 256u; i = i << 1u) {
        workgroupBarrier();
        if wid % (2u * i) == 0u {
            histogram_na[wid + (2u * i) - 1u] += histogram_na[wid + i - 1u];
        }
    }
    workgroupBarrier();
    // special case for first iteration
    if wid < 256u {
        if wid % 256u == 0u {
        // 64 / 2 - 1 = 31
            let before = histogram_na[(256u / 2u) - 1u];

            histogram_na[(256u / 2u) - 1u] = 0u;
            histogram_na[256u - 1u] = before;
        }
    }
    // 32 16 8 4 2
    for (var i: u32 = 256u / 2u; i > 1u; i = i >> 1u) {
        workgroupBarrier();
        if wid < 256u {
            if wid % i == 0u {
                prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
            }
        }
    }
    workgroupBarrier();
}

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn block_offset(wid: u32, digit: u32, id: u32) {
        // compute __ballot with workgroup instructions
    let index = wid / 32u;
    let bit_index = wid % 32u;
    atomicOr(&ballot_mem[index], (digit & 1u) << bit_index);
    workgroupBarrier();
    let temp_buffer: array<u32, 8> = array<u32, 8>(atomicLoad(&ballot_mem[0]), atomicLoad(&ballot_mem[1]), atomicLoad(&ballot_mem[2]), atomicLoad(&ballot_mem[3]), atomicLoad(&ballot_mem[4]), atomicLoad(&ballot_mem[5]), atomicLoad(&ballot_mem[6]), atomicLoad(&ballot_mem[7]));
    if (wid & 1u) != 0u {
        offset_bmp[wid * 8u] = temp_buffer[0] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 1u] = temp_buffer[1] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 2u] = temp_buffer[2] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 3u] = temp_buffer[3] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 4u] = temp_buffer[4] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 5u] = temp_buffer[5] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 6u] = temp_buffer[6] & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 7u] = temp_buffer[7] & 0xFFFFFFFFu;
    } else {
        offset_bmp[wid * 8u] = (~temp_buffer[0]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 1u] = (~temp_buffer[1]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 2u] = (~temp_buffer[2]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 3u] = (~temp_buffer[3]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 4u] = (~temp_buffer[4]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 5u] = (~temp_buffer[5]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 6u] = (~temp_buffer[6]) & 0xFFFFFFFFu;
        offset_bmp[wid * 8u + 7u] = (~temp_buffer[7]) & 0xFFFFFFFFu;
    }
    workgroupBarrier();
    for (var k = 1u; k < 8u; k += 1u) { // log2(256) = 8
        if wid < 8u {
            atomicStore(&ballot_mem[wid], 0u);
        }
        workgroupBarrier();
        // compute __ballot with workgroup instructions
        let index = wid / 32u;
        let bit_index = wid % 32u;
        atomicOr(&ballot_mem[index], ((digit >> k) & 1u) << bit_index);
        workgroupBarrier();
        let temp_buffer: array<u32, 8> = array<u32, 8>(atomicLoad(&ballot_mem[0]), atomicLoad(&ballot_mem[1]), atomicLoad(&ballot_mem[2]), atomicLoad(&ballot_mem[3]), atomicLoad(&ballot_mem[4]), atomicLoad(&ballot_mem[5]), atomicLoad(&ballot_mem[6]), atomicLoad(&ballot_mem[7]));
        if ((wid >> k) & 1u) != 0u {
            offset_bmp[wid * 8u] &= temp_buffer[0];
            offset_bmp[wid * 8u + 1u] &= temp_buffer[1];
            offset_bmp[wid * 8u + 2u] &= temp_buffer[2];
            offset_bmp[wid * 8u + 3u] &= temp_buffer[3];
            offset_bmp[wid * 8u + 4u] &= temp_buffer[4];
            offset_bmp[wid * 8u + 5u] &= temp_buffer[5];
            offset_bmp[wid * 8u + 6u] &= temp_buffer[6];
            offset_bmp[wid * 8u + 7u] &= temp_buffer[7];
        } else {
            offset_bmp[wid * 8u] &= ~temp_buffer[0];
            offset_bmp[wid * 8u + 1u] &= ~temp_buffer[1];
            offset_bmp[wid * 8u + 2u] &= ~temp_buffer[2];
            offset_bmp[wid * 8u + 3u] &= ~temp_buffer[3];
            offset_bmp[wid * 8u + 4u] &= ~temp_buffer[4];
            offset_bmp[wid * 8u + 5u] &= ~temp_buffer[5];
            offset_bmp[wid * 8u + 6u] &= ~temp_buffer[6];
            offset_bmp[wid * 8u + 7u] &= ~temp_buffer[7];
        }
        workgroupBarrier();
    }
    var count: u32 = 0u;
    let full_u32s: u32 = ((wid + 1u) / 32u);
    let remaining_bits = ((wid + 1u) % 32u);
    for (var i = 0u; i < full_u32s; i += 1u) {
        count += countOneBits(offset_bmp[digit * 8u + i]);
    }
    let mask = (1u << remaining_bits) - 1u;
    count += countOneBits(offset_bmp[digit * 8u + full_u32s] & mask);
    local_offset[wid] = count - 1u;
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_index(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;
    let num_blocks_before_this: u32 = id / 256u;
    let num_elements = arrayLength(&codes) / 2u;
    var code_1 = 0u;
    var code_2 = 0u;
    var digit = 16000u;
    if id < num_elements {
        code_1 = codes[id * 2u];
        code_2 = codes[id * 2u + 1u];
        digit = select_digit(code_1, code_2);
        atomicAdd(&histogram[digit], 1u);
    }
    workgroupBarrier();
    histogram_na[wid] = atomicLoad(&histogram[wid]);
    block_offset(wid, digit, id);
    workgroupBarrier();
    prefix_sum_block_exclusive(wid);
    workgroupBarrier();
    if id < num_elements {
        let tri = triangles[wid];
        let histogram_column = num_blocks_before_this;
        let histogram_row = digit;
        let histogram_index = (num_workgroups.x * histogram_row) + histogram_column;
        var prefix_sum_val = histogram_na[digit];
        let local_location = prefix_sum_val + local_offset[wid];
        offset_bmp[6u * local_location] = tri.tri_indices.x;
        offset_bmp[6u * local_location + 1u] = tri.tri_indices.y;
        offset_bmp[6u * local_location + 2u] = tri.tri_indices.z;
        offset_bmp[6u * local_location + 3u] = local_offset[wid] + storage_histograms[histogram_index];
        offset_bmp[6u * local_location + 4u] = code_1;
        offset_bmp[6u * local_location + 5u] = code_2;
    }
    workgroupBarrier();
    if id < num_elements {
        let x = offset_bmp[6u * wid];
        let y = offset_bmp[6u * wid + 1u];
        let z = offset_bmp[6u * wid + 2u];
        let final_location = offset_bmp[6u * wid + 3u];
        let code_1b = offset_bmp[6u * wid + 4u];
        let code_2b = offset_bmp[6u * wid + 5u];
        let tri = TriangleIndices(vec3<u32>(x, y, z));
        codes_2[2u * final_location] = code_1b;
        codes_2[2u * final_location + 1u] = code_2b;
        triangles_2[final_location] = tri;
    }
}
