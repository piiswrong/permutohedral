// #include "cu_hash_table.h"

// namespace mxnet {
// namespace op {
// namespace permutohedral {

// template<int key_size>
// CuHashTable<key_size>::CuHashTable(int32_t n_keys, int32_t *entries, int16_t *keys)
//   : n_keys_(n_keys), entries_(entries), keys_(keys) {
// }

// template<int key_size>
// MSHADOW_FORCE_INLINE __device__ int32_t CuHashTable<key_size>::hash(const int16_t *key) {
//   int32_t h = 0; 
//   for (int32_t i = 0; i < key_size; i++) {
//     h = (h + key[i])* 2531011;
//   }
//   h = h%(2*n_keys_);
//   return h;
// }

// template<int key_size>
// MSHADOW_FORCE_INLINE __device__ int32_t CuHashTable<key_size>::insert(const int16_t *key, int32_t idx) {
//   int32_t h = hash(key);

//   // write our key
//   for (int32_t i = 0; i < key_size; i++) {
//     keys_[idx*key_size+i] = key[i];
//   }

//   while (true) {
//     int32_t *e = entries_ + h;

//     // If the cell is empty (-1), write our key in it.
//     int32_t contents = atomicCAS(e, -1, idx);

//     if (contents == -1) { 
//       // If it was empty, return.
//       return idx;
//     } else {
//       // The cell has a key in it, check if it matches
//       bool match = true;
//       for (int32_t i = 0; i < key_size && match; i++) {
//         match = (keys_[contents*key_size+i] == key[i]);
//       }
//       if (match) return contents;
//     }
//     // increment the bucket with wraparound
//     h++;
//     if (h == n_keys_*2) h = 0;
//   }
// }

// template<int key_size>
// MSHADOW_FORCE_INLINE __device__ int32_t CuHashTable<key_size>::find(const int16_t *key) {
//   int32_t h = hash(key);
//   while (true) {
//     int32_t contents = entries_[h];

//     if (contents == -1) return -1;

//     bool match = true;
//     for (int32_t i = 0; i < key_size && match; i++) {
//         match = (keys_[contents*key_size+i] == key[i]);
//     }
//     if (match) return contents;

//     h++;
//     if (h == n_keys_*2) h = 0;
//   }
// }

// }
// }
// }
