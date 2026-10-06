#define ADDRESS_SUFFIX_FULL_BINS 256u
#define ADDRESS_SUFFIX_FULL_MAX_SUFFIXES 5u
#define ADDRESS_SUFFIX_FULL_FIELDS (ADDRESS_SUFFIX_FULL_BINS * ADDRESS_SUFFIX_FULL_MAX_SUFFIXES)
// One tile pass accumulates a job's three RAF lanes (one flag) and three
// suffix slots: 1536 deferred sums = 30 KiB of threadgroup memory. Keep in
// sync with TILE_FIELDS in address_sequence/mod.rs.
#define ADDRESS_PHASE_RAF_FIELDS (ADDRESS_RAF_DIRECT_LANES * ADDRESS_RAF_DIRECT_BINS)
#define ADDRESS_PHASE_SUFFIX_SLOTS 3u
#define ADDRESS_PHASE_HIGH_SLOTS (ADDRESS_SUFFIX_FULL_MAX_SUFFIXES - ADDRESS_PHASE_SUFFIX_SLOTS)
#define ADDRESS_PHASE_TILE_FIELDS \
    (ADDRESS_PHASE_RAF_FIELDS + ADDRESS_PHASE_SUFFIX_SLOTS * ADDRESS_SUFFIX_FULL_BINS)

struct AddressPhaseParams {
    uint suffix_len;
    uint condense;
};

// Rows [start, end) share one RAF flag and one table; rows without a table
// use index `table count`, whose suffix count is zero.
struct AddressPhaseJob {
    uint start;
    uint end;
    uint table;
    uint raf_flag;
};

struct AddressSuffixFullTable {
    uint job_start;
    uint job_end;
    uint output_start;
    uint suffix_count;
};

struct AddressSuffixFullLookup {
    ulong2 limbs;
};

struct AddressSuffixFullBits {
    ulong lo;
    ulong hi;
    ulong x;
    ulong y;
    uint len;
    uint operand_len;
};

inline ulong address_suffix_full_compact_even_bits(ulong value) {
    value &= 0x5555555555555555ul;
    value = (value | (value >> 1)) & 0x3333333333333333ul;
    value = (value | (value >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    value = (value | (value >> 4)) & 0x00ff00ff00ff00fful;
    value = (value | (value >> 8)) & 0x0000ffff0000fffful;
    value = (value | (value >> 16)) & 0x00000000fffffffful;
    return value;
}

inline ulong address_suffix_full_mask(uint bits) {
    return bits == 0 ? 0ul : ((1ul << bits) - 1ul);
}

inline AddressSuffixFullBits address_suffix_full_bits(
    AddressSuffixFullLookup lookup,
    uint suffix_len)
{
    AddressSuffixFullBits bits;
    bits.len = suffix_len;
    bits.operand_len = suffix_len / 2;
    bits.lo = 0;
    bits.hi = 0;
    if (suffix_len == 64) {
        bits.lo = lookup.limbs[0];
    } else if (suffix_len > 64) {
        bits.lo = lookup.limbs[0];
        bits.hi = lookup.limbs[1] & address_suffix_full_mask(suffix_len - 64);
    } else if (suffix_len != 0) {
        bits.lo = lookup.limbs[0] & address_suffix_full_mask(suffix_len);
    }
    bits.x = address_suffix_full_compact_even_bits(bits.lo >> 1)
        | (address_suffix_full_compact_even_bits(bits.hi >> 1) << 32);
    bits.y = address_suffix_full_compact_even_bits(bits.lo)
        | (address_suffix_full_compact_even_bits(bits.hi) << 32);
    return bits;
}

inline uint address_suffix_full_lookup_byte(AddressSuffixFullLookup lookup, uint shift) {
    return shift < 64
        ? (uint)(lookup.limbs[0] >> shift) & 0xffu
        : (uint)(lookup.limbs[1] >> (shift - 64)) & 0xffu;
}

inline uint address_suffix_full_trailing_zeros(ulong value, uint len) {
    return value == 0 ? len : min((uint)ctz(value), len);
}

inline uint address_suffix_full_leading_ones(ulong value, uint len) {
    if (len == 0) {
        return 0;
    }
    ulong inverse = (~value) & address_suffix_full_mask(len);
    return inverse == 0 ? len : (uint)clz(inverse << (64 - len));
}

inline ulong address_suffix_full_unbounded_shl(ulong value, uint shift) {
    return shift >= 64 ? 0ul : value << shift;
}

inline ulong address_suffix_full_unbounded_shr(ulong value, uint shift) {
    return shift >= 64 ? 0ul : value >> shift;
}

inline ulong address_suffix_full_rotate_right(ulong value, uint shift) {
    return (value >> shift) | (value << (64 - shift));
}

inline uint address_suffix_full_rotate_right_32(uint value, uint shift) {
    return (value >> shift) | (value << (32 - shift));
}

inline uint address_suffix_full_swap_bytes_32(uint value) {
    return ((value & 0x000000ffu) << 24)
        | ((value & 0x0000ff00u) << 8)
        | ((value & 0x00ff0000u) >> 8)
        | ((value & 0xff000000u) >> 24);
}

inline ulong address_suffix_full_pext(ulong x, ulong y) {
    ulong output = 0;
    uint destination = 0;
    while (y != 0) {
        uint source = ctz(y);
        output |= ((x >> source) & 1ul) << destination;
        destination += 1;
        y &= y - 1;
    }
    return output;
}

inline ulong address_suffix_full_window_sign(ulong x, ulong y) {
    return y == 0 ? 0ul : ((x >> (63u - (uint)clz(y))) & 1ul);
}

inline ulong address_suffix_full_sign_extension_w(AddressSuffixFullBits bits) {
    if (bits.len == 0) {
        return 0ul;
    }

    constexpr uint word_half = 32u;
    uint count = min(bits.operand_len, word_half);
    ulong fill = 0;
    if (bits.len >= 64) {
        if (((bits.x >> (word_half - 1)) & 1ul) == 0) {
            return 0ul;
        }
        fill = 0xffffffff00000000ul;
    }
    uint first_position = word_half - count;
    for (uint offset = 0; offset < count; offset++) {
        uint position = first_position + offset;
        if (position != 0) {
            ulong y_bit = (bits.y >> (count - 1 - offset)) & 1ul;
            fill += (1ul - y_bit) << position;
        }
    }
    return fill;
}

// ShiftData{B,H,W}: the suffix-owned low lane bits of `x`, shifted by the
// byte offset the suffix owns from `y` (`eighths` = lane width in eighths of
// the word). Mirrors `shift_data_suffix` in jolt-lookup-tables.
inline ulong address_suffix_full_shift_data(AddressSuffixFullBits bits, uint eighths) {
    uint lane_bits = eighths * 8u;
    ulong lane = lane_bits >= 64u ? bits.x : (bits.x & address_suffix_full_mask(lane_bits));
    uint offset = (uint)(bits.y & (ulong)(8u - eighths));
    return lane << (8u * offset);
}

// OffsetScale{B,H,W}: `2^(8 * offset)` over the offset bits the suffix owns
// (bit `y_i` sits at interleaved position `2i`). Mirrors `offset_scale_suffix`.
inline ulong address_suffix_full_offset_scale(AddressSuffixFullBits bits, uint eighths) {
    uint mask = 8u - eighths;
    uint shift = 0u;
    for (uint i = 0; i < 3u; i++) {
        if (((mask >> i) & 1u) == 1u && 2u * i < bits.len && ((bits.lo >> (2u * i)) & 1ul) == 1ul) {
            shift += 8u << i;
        }
    }
    return 1ul << shift;
}

inline ulong address_suffix_full_evaluate(uchar kind, AddressSuffixFullBits bits) {
    ulong operand_mask = address_suffix_full_mask(bits.operand_len);
    switch (kind) {
        case 0: return 1ul;
        case 1: return bits.x & bits.y;
        case 2: return bits.x & ~bits.y;
        case 3: return bits.x ^ bits.y;
        case 4: return bits.x | bits.y;
        case 5: return bits.y;
        case 6: return (ulong)(uint)bits.y;
        case 7: return (ulong)(bits.x == 0 && bits.y == operand_mask);
        case 8: {
            uint len = min(bits.operand_len, 32u);
            return (ulong)((uint)bits.x == 0 && (uint)bits.y == (uint)address_suffix_full_mask(len));
        }
        case 9: return bits.hi;
        case 10: return bits.lo;
        case 11: return (ulong)(uint)bits.lo;
        case 12: return (ulong)(bits.x < bits.y);
        case 13: return (ulong)(bits.x > bits.y);
        case 14: return (ulong)(bits.x == bits.y);
        case 15: return (ulong)(bits.x == 0);
        case 16: return (ulong)(bits.y == 0);
        case 17: return bits.len == 0 ? 1ul : bits.lo & 1ul;
        case 18: return (ulong)(bits.x == 0 && bits.y == operand_mask);
        case 19: return bits.len == 0 ? 1ul : 1ul << (bits.lo & 63ul);
        case 20: return bits.len == 0 ? 1ul : 1ul << (bits.lo & 31ul);
        case 21: {
            ulong lo = (ulong)address_suffix_full_swap_bytes_32((uint)bits.lo);
            ulong hi = (ulong)address_suffix_full_swap_bytes_32((uint)(bits.lo >> 32));
            return lo | (hi << 32);
        }
        case 22: return bits.len == 0 ? 1ul : 1ul << (63u - (uint)(bits.lo & 63ul));
        case 23: return address_suffix_full_unbounded_shr(
            bits.x,
            address_suffix_full_trailing_zeros(bits.y, bits.operand_len));
        case 24: return 1ul << address_suffix_full_leading_ones(bits.y, bits.operand_len);
        case 25: {
            uint padding = address_suffix_full_trailing_zeros(bits.y, bits.operand_len);
            return padding == 0 ? 0ul : (~0ul << (64 - padding));
        }
        case 26: return address_suffix_full_unbounded_shl(
            bits.x & ~bits.y,
            address_suffix_full_leading_ones(bits.y, bits.operand_len));
        case 27: return (ulong)(bits.len == 0 || (bits.lo & 3ul) == 0);
        case 28: {
            if (bits.len < 32) return 1ul;
            return ((bits.lo >> 31) & 1ul) != 0 ? 0xffffffff00000000ul : 0ul;
        }
        case 29: {
            if (bits.len < 64) return 1ul;
            return ((bits.lo >> 62) & 1ul) != 0 ? 0xffffffff00000000ul : 0ul;
        }
        case 30: {
            uint shift = min(address_suffix_full_trailing_zeros(bits.y, bits.operand_len), 32u);
            return shift == 32 ? 0ul : (ulong)((uint)bits.x >> shift);
        }
        case 31: {
            uint len = min(bits.operand_len, 32u);
            return 1ul << address_suffix_full_leading_ones((uint)bits.y, len);
        }
        case 32: {
            uint leading = address_suffix_full_leading_ones((uint)bits.y, bits.operand_len);
            return leading >= 32 ? 0ul : (ulong)(1u << leading);
        }
        case 33: {
            uint len = min(bits.operand_len, 32u);
            uint leading = address_suffix_full_leading_ones((uint)bits.y, len);
            uint value = (uint)bits.x & ~(uint)bits.y;
            return leading >= 32 ? 0ul : (ulong)(value << leading);
        }
        case 34: return (ulong)(bits.hi == 0);
        case 35: return address_suffix_full_rotate_right(bits.x ^ bits.y, 16);
        case 36: return address_suffix_full_rotate_right(bits.x ^ bits.y, 24);
        case 37: return address_suffix_full_rotate_right(bits.x ^ bits.y, 32);
        case 38: return address_suffix_full_rotate_right(bits.x ^ bits.y, 63);
        case 39: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 16);
        case 40: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 12);
        case 41: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 8);
        case 42: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 7);
        case 43: {
            if (bits.len < 3) return 1ul;
            return 1ul << (((bits.lo >> 2) & 1ul) * 32);
        }
        case 44: return address_suffix_full_pext(bits.x, bits.y);
        case 45: {
            uint count = popcount(bits.y);
            return count < 64 ? 1ul << count : 0ul;
        }
        case 46: return address_suffix_full_window_sign(bits.x, bits.y);
        case 47: {
            ulong sign = address_suffix_full_window_sign(bits.x, bits.y);
            uint count = popcount(bits.y);
            return sign != 0 && count < 64 ? 1ul << count : 0ul;
        }
        case 48: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 22);
        case 49: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 19);
        case 50: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 6);
        case 51: return address_suffix_full_sign_extension_w(bits);
        case 52: return bits.len < 64 ? 0ul : ((bits.x >> 31) & 1ul) * (bits.y & 1ul);
        case 53: return 1ul << (8 * (uint)(bits.lo & 7ul));
        case 54: return 1ul << (8 * (uint)(bits.lo & 6ul));
        case 55: return bits.lo & ~7ul;
        case 56: return address_suffix_full_shift_data(bits, 1u);
        case 57: return address_suffix_full_shift_data(bits, 2u);
        case 58: return address_suffix_full_shift_data(bits, 4u);
        case 59: return address_suffix_full_offset_scale(bits, 1u);
        case 60: return address_suffix_full_offset_scale(bits, 2u);
        case 61: return address_suffix_full_offset_scale(bits, 4u);
        case 62: {
            uint pairs = bits.len / 2u;
            if (pairs == 64u) {
                return bits.x ^ ((bits.y << 1) | (bits.y >> 63));
            }
            return (bits.x ^ (bits.y << 1)) & address_suffix_full_mask(pairs) & ~1ul;
        }
        case 63: {
            uint pairs = bits.len / 2u;
            return pairs == 0u ? 0ul : (bits.y >> (pairs - 1u)) & 1ul;
        }
        case 64: return bits.len == 0u ? 0ul : (bits.x & 1ul);
        default: return 0ul;
    }
}

inline SolinasFp128 address_suffix_full_field_from_u128(ulong lo, ulong hi) {
    SolinasFp128 value;
    value.limb = uint4((uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32));
    return value;
}

// Suffix slots [FIRST, FIRST + SLOTS) of one row. The fixed bounds allow
// unrolling.
template <uint FIRST, uint SLOTS>
inline void address_phase_add_suffixes(
    threadgroup atomic_uint* sums,
    thread SolinasLazySum (&chunk_zero)[SLOTS],
    device const uchar* kinds,
    uint suffix_count,
    uint first_field,
    AddressSuffixFullBits bits,
    uint chunk,
    SolinasFp128 weight)
{
    for (uint slot = 0; slot < SLOTS; slot++) {
        if (FIRST + slot >= suffix_count) {
            break;
        }
        ulong scalar = address_suffix_full_evaluate(kinds[FIRST + slot], bits);
        if (scalar == 0) {
            continue;
        }
        SolinasFp128 contribution = scalar == 1
            ? weight
            : solinas_mul_wide(weight, address_suffix_full_field_from_u128(scalar, 0));
        if (chunk == 0) {
            solinas_lazy_add(chunk_zero[slot], contribution);
        } else {
            solinas_deferred_atomic_add_5(
                sums, first_field + slot * ADDRESS_SUFFIX_FULL_BINS + chunk, contribution);
        }
    }
}

template <uint FIRST, uint SLOTS>
inline void address_phase_flush_suffixes(
    threadgroup atomic_uint* sums,
    thread SolinasLazySum (&chunk_zero)[SLOTS],
    uint suffix_count,
    uint first_field,
    uint lane)
{
    for (uint slot = 0; slot < SLOTS; slot++) {
        if (FIRST + slot < suffix_count) {
            solinas_deferred_atomic_flush_simd(
                sums, first_field + slot * ADDRESS_SUFFIX_FULL_BINS, chunk_zero[slot], lane);
        }
    }
}

// One read of each row feeds its job's RAF sums and suffix slots 0-2.
kernel void solinas_address_phase_tile(
    device const AddressSuffixFullLookup* lookups [[buffer(0)]],
    device SolinasFp128* weights [[buffer(1)]],
    device const SolinasFp128* previous_phase_table [[buffer(2)]],
    device const AddressPhaseJob* jobs [[buffer(3)]],
    device const uchar* suffix_kinds [[buffer(4)]],
    device const uchar* suffix_counts [[buffer(5)]],
    device SolinasFp128* raf_partials [[buffer(6)]],
    device SolinasFp128* suffix_partials [[buffer(7)]],
    constant AddressPhaseParams& params [[buffer(8)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint job_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    for (uint counter = tid; counter < ADDRESS_PHASE_TILE_FIELDS * SOLINAS_DEFERRED_SUM_WORDS;
         counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AddressPhaseJob job = jobs[job_index];
    uint suffix_count = suffix_counts[job.table];
    device const uchar* kinds = suffix_kinds + job.table * ADDRESS_SUFFIX_FULL_MAX_SUFFIXES;
    uint upper_bits = params.suffix_len > 64 ? params.suffix_len - 64 : 0;
    // Chunk 0 dominates (padding rows, the high chunks of 64-bit lookups);
    // its fields are summed per thread instead of contended atomics.
    SolinasLazySum raf_zero[ADDRESS_RAF_DIRECT_LANES];
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
        raf_zero[output_lane] = solinas_lazy_zero();
    }
    SolinasLazySum suffix_zero[ADDRESS_PHASE_SUFFIX_SLOTS];
    for (uint slot = 0; slot < ADDRESS_PHASE_SUFFIX_SLOTS; slot++) {
        suffix_zero[slot] = solinas_lazy_zero();
    }
    for (uint row = job.start + tid; row < job.end; row += threads) {
        AddressSuffixFullLookup lookup = lookups[row];
        SolinasFp128 weight = weights[row];
        if (params.condense != 0) {
            uint previous_chunk = address_suffix_full_lookup_byte(lookup, params.suffix_len + 8);
            weight = solinas_mul_wide(weight, previous_phase_table[previous_chunk]);
            weights[row] = weight;
        }
        AddressSuffixFullBits bits = address_suffix_full_bits(lookup, params.suffix_len);
        uint chunk = address_suffix_full_lookup_byte(lookup, params.suffix_len);

        SolinasFp128 raf[ADDRESS_RAF_DIRECT_LANES];
        bool present[ADDRESS_RAF_DIRECT_LANES];
        raf[0] = weight;
        present[0] = true;
        if (job.raf_flag == 0) {
            present[1] = bits.x != 0;
            present[2] = bits.y != 0;
            raf[1] = present[1]
                ? solinas_mul_wide(weight, address_suffix_full_field_from_u128(bits.x, 0))
                : solinas_zero();
            raf[2] = present[2]
                ? solinas_mul_wide(weight, address_suffix_full_field_from_u128(bits.y, 0))
                : solinas_zero();
        } else {
            present[1] = bits.lo != 0 || bits.hi != 0;
            present[2] = upper_bits == 0 || bits.hi == address_suffix_full_mask(upper_bits);
            raf[1] = present[1]
                ? solinas_mul_wide(weight, address_suffix_full_field_from_u128(bits.lo, bits.hi))
                : solinas_zero();
            raf[2] = weight;
        }
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            if (!present[output_lane]) {
                continue;
            }
            if (chunk == 0) {
                solinas_lazy_add(raf_zero[output_lane], raf[output_lane]);
            } else {
                solinas_deferred_atomic_add_5(
                    sums, chunk * ADDRESS_RAF_DIRECT_LANES + output_lane, raf[output_lane]);
            }
        }
        address_phase_add_suffixes<0, ADDRESS_PHASE_SUFFIX_SLOTS>(
            sums, suffix_zero, kinds, suffix_count, ADDRESS_PHASE_RAF_FIELDS, bits, chunk, weight);
    }
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
        solinas_deferred_atomic_flush_simd(sums, output_lane, raf_zero[output_lane], lane);
    }
    address_phase_flush_suffixes<0, ADDRESS_PHASE_SUFFIX_SLOTS>(
        sums, suffix_zero, suffix_count, ADDRESS_PHASE_RAF_FIELDS, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The RAF finalize sums whole [job][flag][chunk][lane] blocks, so the
    // other flag's half is written as zero.
    uint raf_base = job_index * ADDRESS_RAF_DIRECT_FIELDS;
    uint own_half = job.raf_flag * ADDRESS_PHASE_RAF_FIELDS;
    uint other_half = ADDRESS_PHASE_RAF_FIELDS - own_half;
    for (uint field = tid; field < ADDRESS_PHASE_RAF_FIELDS; field += threads) {
        raf_partials[raf_base + own_half + field] = solinas_deferred_atomic_reduce_5(sums, field);
        raf_partials[raf_base + other_half + field] = solinas_zero();
    }
    uint suffix_base = job_index * ADDRESS_SUFFIX_FULL_FIELDS;
    uint low_fields = min(suffix_count, ADDRESS_PHASE_SUFFIX_SLOTS) * ADDRESS_SUFFIX_FULL_BINS;
    for (uint field = tid; field < low_fields; field += threads) {
        suffix_partials[suffix_base + field] =
            solinas_deferred_atomic_reduce_5(sums, ADDRESS_PHASE_RAF_FIELDS + field);
    }
    if (suffix_count <= ADDRESS_PHASE_SUFFIX_SLOTS) {
        return;
    }

    // Tables with more suffixes than one pass holds re-read their rows (and
    // this thread's condensed weights) for the remaining slots.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint counter = tid;
         counter < ADDRESS_PHASE_HIGH_SLOTS * ADDRESS_SUFFIX_FULL_BINS * SOLINAS_DEFERRED_SUM_WORDS;
         counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SolinasLazySum high_zero[ADDRESS_PHASE_HIGH_SLOTS];
    for (uint slot = 0; slot < ADDRESS_PHASE_HIGH_SLOTS; slot++) {
        high_zero[slot] = solinas_lazy_zero();
    }
    for (uint row = job.start + tid; row < job.end; row += threads) {
        AddressSuffixFullLookup lookup = lookups[row];
        address_phase_add_suffixes<ADDRESS_PHASE_SUFFIX_SLOTS, ADDRESS_PHASE_HIGH_SLOTS>(
            sums,
            high_zero,
            kinds,
            suffix_count,
            0,
            address_suffix_full_bits(lookup, params.suffix_len),
            address_suffix_full_lookup_byte(lookup, params.suffix_len),
            weights[row]);
    }
    address_phase_flush_suffixes<ADDRESS_PHASE_SUFFIX_SLOTS, ADDRESS_PHASE_HIGH_SLOTS>(
        sums, high_zero, suffix_count, 0, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint high_fields = (suffix_count - ADDRESS_PHASE_SUFFIX_SLOTS) * ADDRESS_SUFFIX_FULL_BINS;
    for (uint field = tid; field < high_fields; field += threads) {
        suffix_partials[suffix_base + low_fields + field] =
            solinas_deferred_atomic_reduce_5(sums, field);
    }
}

kernel void solinas_address_suffix_full_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device const AddressSuffixFullTable* tables [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    uint table [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    AddressSuffixFullTable descriptor = tables[table];
    uint fields = descriptor.suffix_count * ADDRESS_SUFFIX_FULL_BINS;
    for (uint field = tid; field < fields; field += threads) {
        SolinasFp128 sum = solinas_zero();
        for (uint job = descriptor.job_start; job < descriptor.job_end; job++) {
            sum = solinas_add(sum, partials[job * ADDRESS_SUFFIX_FULL_FIELDS + field]);
        }
        uint suffix = field / ADDRESS_SUFFIX_FULL_BINS;
        uint chunk = field & (ADDRESS_SUFFIX_FULL_BINS - 1);
        output[(descriptor.output_start + suffix) * ADDRESS_SUFFIX_FULL_BINS + chunk] = sum;
    }
}
