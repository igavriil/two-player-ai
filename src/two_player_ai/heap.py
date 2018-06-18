import numpy as np
import numba as nb


class Heap(object):
    def __init__(self, size):
        self.heap = np.zeros(size)
        self.heap.fill(np.nan)
        self.count = 0

    def push_item(self, item):
        self.count = Heap.push(self.heap, item, self.count)

    def pop_item(self):
        if self.count == 0:
            item = None
        else:
            item, self.count = Heap.pop(self.heap, self.count)

        return item

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def pop(heap, count):
        root = heap[0]
        count -= 1

        heap[0] = heap[count]
        heap[count] = np.nan

        parent_id = 0
        left_child_id = 2 * parent_id + 1
        right_child_id = 2 * parent_id + 2

        while (heap[parent_id] > heap[left_child_id] or
               heap[parent_id] > heap[right_child_id]):
            if heap[right_child_id] < heap[left_child_id]:
                smallest_child_id = right_child_id
            else:
                smallest_child_id = left_child_id

            temp = heap[smallest_child_id]
            heap[smallest_child_id] = heap[parent_id]
            heap[parent_id] = temp

            parent_id = smallest_child_id
            left_child_id = 2 * parent_id + 1
            right_child_id = 2 * parent_id + 2
            if left_child_id >= count or right_child_id >= count:
                break

        return root, count

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def push(heap, value, count):

        heap[count] = value
        child_id = count
        if child_id % 2 == 0:
            parent_id = int(child_id / 2 - 1)
        else:
            parent_id = int(np.floor(child_id / 2))

        while heap[child_id] < heap[parent_id]:
            temp = heap[parent_id]
            heap[parent_id] = heap[child_id]
            heap[child_id] = temp
            child_id = parent_id
            if child_id % 2 == 0:
                parent_id = int(child_id / 2 - 1)
            else:
                parent_id = int(np.floor(child_id / 2))

        count += 1

        return count
