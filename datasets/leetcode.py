import numpy as np
import pdb
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None



class Solution:
    # @param s, a string
    # @return a string
    def reverseWords(self, s):
        print(s.split())
        return " ".join(s.split()[::-1])

    def maxProduct(self, A):
        global_max, local_max, local_min = float("-inf"), 1, 1
        for x in A:
            local_max = max(1, local_max)
            if x > 0:
                local_max, local_min = local_max * x, local_min * x
            else:
                local_max, local_min = local_min * x, local_max * x
            global_max = max(global_max, local_max)
        return global_max

    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        result, dvd = "", n

        while dvd:
            result += chr((dvd - 1) % 26 + ord('A'))
            print ord('A')
            dvd = (dvd - 1) / 26

        return result[::-1]



    def majorityElement(self, num):
        d = {}
        l = len(num)
        for i in num:
            if d.has_key(i):
                d[i] += 1
                if d[i] > l/2:
                    return i
            else:
                d[i] = 1
                if d[i] > l/2:
                    return i


    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = 0
        for i in xrange(len(s)):
            result *= 26
            result += ord(s[i]) - ord('A') + 1
        return result

    def reverseBits(self, n):
        result = 0
        for i in xrange(32):
            result <<= 1
            result |= n & 1
            # result= ((result or n) and 1)

            n >>= 1
        return result


    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return intervals
        intervals.sort(key=lambda x: x.start)
        result = [intervals[0]]
        for i in xrange(1, len(intervals)):
            prev, current = result[-1], intervals[i]
            if current.start <= prev.end:
                prev.end = max(prev.end, current.end)
            else:
                result.append(current)
        return result

    def lengthOfLastWord(self, s):
        s1=s.split()
        print(len(s1[-1]))

    def generateMatrix(self, n):
        if n == 0: return []
        matrix = [[0 for i in range(n)] for j in range(n)]
        up = 0; down = len(matrix)-1
        left = 0; right = len(matrix[0])-1
        direct = 0; count = 0
        while True:
            if direct == 0:
                for i in range(left, right+1):
                    count += 1; matrix[up][i] = count
                up += 1
            if direct == 1:
                for i in range(up, down+1):
                    count += 1; matrix[i][right] = count
                right -= 1
            if direct == 2:
                for i in range(right, left-1, -1):
                    count += 1; matrix[down][i] = count
                down -= 1
            if direct == 3:
                for i in range(down, up-1, -1):
                    count += 1; matrix[i][left] = count
                left += 1
            if count == n*n: return matrix
            direct = (direct+1) % 4

    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head or not head.next:
            return head

        n, cur = 1, head
        while cur.next:
            cur = cur.next
            n += 1
        cur.next = head

        cur, tail = head, cur
        for _ in xrange(n - k % n):
            tail = cur
            cur = cur.next
        tail.next = None

        return cur


    def sortColors(self, A):
        if A == []: return
        p0 = 0; p2 = len(A) - 1; i = 0
        while i <= p2:
            if A[i] == 2:
                A[i], A[p2] = A[p2], A[i]
                p2 -= 1
            elif A[i] == 0:
                A[i], A[p0] = A[p0], A[i]
                p0 += 1
                i += 1
            else:
                i += 1


    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(-1)
        dummy.next = head
        curr = dummy
        is_repeat = False
        while curr.next:
            while curr.next.next and curr.next.val == curr.next.next.val:
                curr.next = curr.next.next
                is_repeat = True
            if is_repeat:
                curr.next = curr.next.next
                is_repeat = False
            else:
                curr = curr.next
        return dummy.next


    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        for num in sorted(nums):
            result += [item + [num] for item in result]
                # result += item + [num]

        return result

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        nums.sort()
        temp_size = 0
        for i in range(len(nums)):
            start = temp_size
            temp_size = len(result)
            # for j in range(start, temp_size):
            for j in range(temp_size):

                result.append(result[j] + [nums[i]])
        return result


if __name__=="__main__":
    s = "the sky is blue"
    # print Solution().reverseWords(s)
    # print Solution().convertToTitle(30)
    # print Solution().reverseBits(50)
    # print Solution().lengthOfLastWord("Hello World    eeeeeeee    ")
    print Solution().subsetsWithDup([1, 2, 3])
    print Solution().subsets([1, 2, 3 ,2 ])
