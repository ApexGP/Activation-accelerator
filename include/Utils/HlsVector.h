#ifndef HLS_VECTOR_H
#define HLS_VECTOR_H

#include <array>      // HLS-friendly fixed-size array
#include <cstddef>    // For size_t
#include <stdexcept>  // For std::out_of_range

#ifndef __SYNTHESIS__
// Include assert.h only during C++ simulation (ignored during synthesis)
#include <cassert>
#define HLS_ASSERT(condition, message) assert((condition) && (message))
#else
// During synthesis, assert is a no-op
#define HLS_ASSERT(condition, message)
#endif

/**
 * @brief HLS-compatible vector-like container.
 * * Simulates std::vector interface but uses a fixed-capacity std::array as 
 * the backend storage.
 * * All memory is determined at compile-time; no dynamic memory allocation.
 * *
 * @tparam T Data type
 * @tparam MaxCapacity The fixed maximum capacity determined at compile-time
 */
template <typename T, size_t MaxCapacity>
class HlsVector
{
public:
    // Type aliases
    using value_type = T;
    using size_type = size_t;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

private:
    // Core: Use std::array as the backend storage for HLS
    std::array<T, MaxCapacity> _data;

    // Tracks the current number of elements in use
    size_type _size;

public:
    // --- Constructors ---
    HlsVector() : _data(), _size(0)
    {
// Hint to HLS that this is a data structure and all members should be initialized
#pragma HLS INLINE
    }

    // --- 1. size() ---
    size_type size() const
    {
#pragma HLS INLINE
        return _size;
    }

    // --- 2. empty() ---
    bool empty() const
    {
#pragma HLS INLINE
        return _size == 0;
    }

    // --- 3. begin() / end() ---
    // Returns a pointer/iterator to the internal std::array
    pointer begin()
    {
#pragma HLS INLINE
        return _data.data();
    }

    const_pointer begin() const
    {
#pragma HLS INLINE
        return _data.data();
    }

    // Key: end() points to the position *after* the last valid element
    pointer end()
    {
#pragma HLS INLINE
        return _data.data() + _size;
    }

    const_pointer end() const
    {
#pragma HLS INLINE
        return _data.data() + _size;
    }

    // --- 4. push_back() ---
    /**
     * @brief Appends an element to the end.
     * HLS WARNING: If _size >= MaxCapacity, this operation will fail.
     * In C simulation, this will trigger an assert.
     * In synthesized hardware, it will fail silently (element not added).
     */
    void push_back(const_reference value)
    {
#pragma HLS INLINE

        // Check for overflow during C simulation
        HLS_ASSERT(_size < MaxCapacity, "HlsVector push_back overflow");

        // HLS Synthesis: Only write if not full
        if (_size < MaxCapacity) {
            _data[_size] = value;
            _size++;
        }
    }

    // --- 5. resize() / reserve() ---

    /**
     * @brief !! HLS Limitation !!
     * reserve() is meaningless in this model as capacity is fixed.
     * Only provide capacity() to query the maximum size. No reserve() function is provided.
     */
    size_type capacity() const
    {
#pragma HLS INLINE
        return MaxCapacity;
    }

    /**
     * @brief Resizes the HlsVector.
     * @param new_size The new size.
     * @param value Default value to fill new elements with if growing.
     */
    void resize(size_type new_size, const_reference value = T())
    {
        if (new_size > MaxCapacity) {
            // HLS ERROR: Cannot exceed maximum capacity.
            // Assert in simulation, clamp to MaxCapacity in synthesis.
            HLS_ASSERT(false, "HlsVector resize exceeds MaxCapacity");
            new_size = MaxCapacity;
        }

        if (new_size > _size) {
            // Growing (fill new elements)
            // HLS will unroll or pipeline this loop
            for (size_type i = _size; i < new_size; ++i) {
#pragma HLS PIPELINE
                _data[i] = value;
            }
        }
        // Both growing and shrinking require updating _size
        _size = new_size;
    }

    // --- 6. operator[] ---
    reference operator[](size_type index)
    {
#pragma HLS INLINE
        // For HLS synthesis, we rely on the caller to ensure index < _size
        // Add assert for C simulation
        HLS_ASSERT(index < _size, "HlsVector index out of range");
        return _data[index];
    }

    const_reference operator[](size_type index) const
    {
#pragma HLS INLINE
        HLS_ASSERT(index < _size, "HlsVector index out of range");
        return _data[index];
    }

    // --- 7. operator= (Copy Assignment) ---
    HlsVector &operator=(const HlsVector &other)
    {
#pragma HLS INLINE
        if (this == &other) {
            return *this;
        }

        // Ensure HLS understands this is a data copy
        // HLS will unroll this loop
        for (size_type i = 0; i < other._size; ++i) {
#pragma HLS UNROLL
            _data[i] = other._data[i];
        }

        _size = other._size;

        return *this;
    }

    // (Copy Constructor - usually default-generated, but good to be explicit)
    HlsVector(const HlsVector &other)
    {
#pragma HLS INLINE
        *this = other;  // Call the copy assignment operator
    }

    // --- Other useful functions ---
    void clear()
    {
#pragma HLS INLINE
        _size = 0;
    }

    reference front()
    {
#pragma HLS INLINE
        HLS_ASSERT(_size > 0, "front() called on empty HlsVector");
        return _data[0];
    }

    reference back()
    {
#pragma HLS INLINE
        HLS_ASSERT(_size > 0, "back() called on empty HlsVector");
        return _data[_size - 1];
    }

};  // end class HlsVector

#endif  // HLS_VECTOR_H