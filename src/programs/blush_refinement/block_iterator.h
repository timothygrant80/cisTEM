#ifndef _SRC_PROGRAMS_BLUSH_REFINEMENT_BLOCK_ITERATOR_H_
#define _SRC_PROGRAMS_BLUSH_REFINEMENT_BLOCK_ITERATOR_H_

/**
 * @brief Class enabling iteration over subsets of a volume.
 * Theoretically useful for 1D and 2D objects as well. It handles the state
 * of the container and manages ranges.
 * 
 */
class BlockIterator {
  public:
    BlockIterator(const std::tuple<int, int, int>& dims, const int& block_size, const int& strides) {
        x_range = GetStartingIndices(std::get<0>(dims), block_size, strides);
        y_range = GetStartingIndices(std::get<1>(dims), block_size, strides);
        z_range = GetStartingIndices(std::get<2>(dims), block_size, strides);
    }

    /**
     * @brief Encapsulates current state of the iteration (i.e. the current block)
     * and controls movement to next block.
     * 
     */
    class Iterator {
      public:
        Iterator(BlockIterator& parent, bool is_end = false) : parent(parent), x_index(0), y_index(0), z_index(0), index(0), is_end(is_end){ };

        // Sort of backward logic; want this to return false when reaching the end, not true; so, we invert
        // is_end within this check.
        bool operator!=(const Iterator& other) {
            return ! is_end;
        }

        std::tuple<int, int, int> operator*( ) {
            return {parent.x_range[x_index], parent.y_range[y_index], parent.z_range[z_index]};
        }

        int size( ) const {
            return parent.x_range.size( ) * parent.y_range.size( ) * parent.z_range.size( );
        }

        Iterator& operator++( ) {
            x_index++;
            if ( x_index == parent.x_range.size( ) ) {
                x_index = 0;
                y_index++;
            }

            if ( y_index == parent.y_range.size( ) ) {
                y_index = 0;
                z_index++;
            }

            if ( z_index == parent.z_range.size( ) )
                is_end = true;
            else
                index++;

            return *this;
        }

      private:
        BlockIterator& parent;
        int            x_index, y_index, z_index, index;
        bool           is_end;
    };

    // Begin and end iteration; begin is called when starting the range-based for loop,
    // while end should be called when the loop is complete
    Iterator begin( ) {
        return Iterator(*this, false);
    }

    Iterator end( ) {
        return Iterator(*this, true);
    }

    std::tuple<int, int, int> get_coords_by_index(int index) {
        int y_size = y_range.size( );
        int x_size = x_range.size( );

        int z_idx     = index / (x_size * y_size);
        int remaining = index % (x_size * y_size);
        int y_idx     = remaining / x_size;
        int x_idx     = remaining % x_size;

        return {x_range[x_idx], y_range[y_idx], z_range[z_idx]};
    }

    /**
     * @brief Get the next coords to be used for slices that will be passed to Blush.
     * 
     * @return std::tuple<int, int, int> Contains the coordinates to be used in blocks. If these integers are negative, iterating will cease.
     */
    // std::tuple<int, int, int> next( ) {
    //     if ( ! has_next( ) ) {
    //         return {-1, -1, -1};
    //     }

    //     if ( x_index == x_range.size( ) ) {
    //         x_index = 0;
    //         y_index++;
    //     }
    //     if ( y_index == y_range.size( ) ) {
    //         y_index = 0;
    //         z_index++;
    //     }

    //     std::tuple<int, int, int> coords{x_range[x_index], y_range[y_index], z_range[z_index]};
    //     x_index++;
    //     index++;

    //     return coords;
    // }

  private:
    std::vector<int> x_range, y_range, z_range;

    /**
     * @brief Based on numpy arange; it creates a vector with the specified
     * endpoint based on the desired block size and dimension. For Blush purposes,
     * it will give the starting indices of the blocks that will be processed through
     * the model.
     * 
     * @param dim Size of the dimension in question (usually x, y, or z).
     * @param block_size Size of the block's dimension (x, y, or z).
     * @param strides Amount by which to increment each element.
     * @return std::vector<int> Contains indices for start of each block.
     */
    std::vector<int> GetStartingIndices(const int& dim, const int& block_size, const int& strides) {
        // Difference between image dim and block dim to determine final index
        int              span    = dim - block_size;
        int              n_steps = (span / strides) + 1;
        std::vector<int> r(n_steps);
        for ( int i = 0; i < n_steps; i++ ) {
            r[i] = i * strides;
            // wxPrintf("r[%i] == %i\n", i, r[i]);
        }

        if ( r.back( ) != span )
            r.push_back(span);
        return r;
    };

    /**
     * @brief Check if this iterator has reached the end of the block.
     * 
     * @return true Reached end.
     * @return false Have not yet reached end.
     */
    // bool has_next( ) {
    //     return z_index < z_range.size( );
    // }
};
#endif