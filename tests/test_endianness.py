
import struct

# Suppose target = 0x00000000ffff0000000000000000000000000000000000000000000000000000
targ_int = 0x00000000ffff0000000000000000000000000000000000000000000000000000
targ_64 = (targ_int >> 192) & 0xFFFFFFFFFFFFFFFF
print(f'targ_64: 0x{targ_64:016x}')

# Suppose a valid hash has 4 leading zero bytes: 00 00 00 00 12 34 56 78 ...
h_bytes = bytes.fromhex('0000000012345678' + '00'*24)
val_u64_le = struct.unpack('<Q', h_bytes[:8])[0]
val_u64_be = struct.unpack('>Q', h_bytes[:8])[0]
print(f'val_u64_le (what ((uint64_t*)h)[0] reads): 0x{val_u64_le:016x}')
print(f'val_u64_be: 0x{val_u64_be:016x}')
print('val_u64_le <= targ_64:', val_u64_le <= targ_64)
print('val_u64_be <= targ_64:', val_u64_be <= targ_64)

# Now suppose an INVALID hash (random bytes, NOT leading zeros in big endian): 87 65 43 21 00 00 00 00 ...
h_invalid = bytes.fromhex('8765432100000000' + 'ff'*24)
val_inv_le = struct.unpack('<Q', h_invalid[:8])[0]
print(f'val_inv_le: 0x{val_inv_le:016x}')
print('val_inv_le <= targ_64:', val_inv_le <= targ_64)
