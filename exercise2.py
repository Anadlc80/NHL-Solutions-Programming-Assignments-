# There are two simple mistakes in this code: 1) you are using wrong number of colum using the column 1 for both columns the 0 and 1.
# And the most important one, you are losing the information of several column when you give them other values. 
# And as we now, we need to use a list of index, because Numpy evaluate first the right side and safe it in the temporal buffer. 
# For example if you introduce in the colum 0 the values of colum 1 you already lost the values of colum 0

def swap(coords: np.ndarray):
        coords[:,[0,1,2,3]]= coords[:,[1,0,3,2]]
    return coords



coords = np.array([[10, 5, 15, 6, 0],
                   [11, 3, 13, 6, 0],
                   [5, 3, 13, 6, 1],
                   [4, 4, 13, 6, 1],
                   [6, 5, 13, 16, 1]])
swapped_coords = swap(coords)
print(swapped_coords)
