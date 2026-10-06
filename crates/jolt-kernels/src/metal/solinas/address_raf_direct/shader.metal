#define ADDRESS_RAF_DIRECT_BINS 256u
#define ADDRESS_RAF_DIRECT_KEYS (2u * ADDRESS_RAF_DIRECT_BINS)
#define ADDRESS_RAF_DIRECT_LANES 3u
#define ADDRESS_RAF_DIRECT_FIELDS (ADDRESS_RAF_DIRECT_KEYS * ADDRESS_RAF_DIRECT_LANES)

inline SolinasFp128 address_direct_simd_sum(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

kernel void solinas_address_raf_direct_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant uint& job_count [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint key [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[ADDRESS_RAF_DIRECT_LANES];
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
        sums[output_lane] = solinas_zero();
    }
    uint field = key * ADDRESS_RAF_DIRECT_LANES;
    for (uint job = tid; job < job_count; job += threads) {
        uint base = job * ADDRESS_RAF_DIRECT_FIELDS + field;
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            sums[output_lane] = solinas_add(sums[output_lane], partials[base + output_lane]);
        }
    }

    uint simdgroups = threads / 32;
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
        SolinasFp128 sum = address_direct_simd_sum(sums[output_lane]);
        if (lane == 0) {
            shared[output_lane * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            SolinasFp128 sum = lane < simdgroups
                ? shared[output_lane * simdgroups + lane]
                : solinas_zero();
            sum = address_direct_simd_sum(sum);
            if (lane == 0) {
                uint chunk = key & (ADDRESS_RAF_DIRECT_BINS - 1);
                uint first_lane = key >= ADDRESS_RAF_DIRECT_BINS ? 3 : 0;
                output[(first_lane + output_lane) * ADDRESS_RAF_DIRECT_BINS + chunk] = sum;
            }
        }
    }
}
