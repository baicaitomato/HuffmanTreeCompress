"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
from utils import *
from huffman import HuffmanTree



# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True

    >>> d = build_frequency_dict(b'Hello world!')
    >>> d
    {72: 1, 101: 1, 108: 3, 111: 2, 32: 1, 119: 1, 114: 1, 100: 1, 33: 1}
    """
    # TODO: Implement this function
    dic = {}
    for i in text:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    return dic


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq == {2: 6, 3: 4}
    True

    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> freq == {2: 6, 3: 4, 7: 5}
    True

    >>> freq = {11: 7, 14: 4, 12: 5, 13: 2}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(11), \
                             HuffmanTree(None, HuffmanTree(12), \
                             HuffmanTree(None, HuffmanTree(13), \
                             HuffmanTree(14))))
    >>> t == result
    True
    >>> freq == {11: 7, 14: 4, 12: 5, 13: 2}
    True

    >>> freq = {5: 5, 6: 6, 7: 7, 8: 8, 15: 15}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(None, \
                             HuffmanTree(7), HuffmanTree(8)), \
                             HuffmanTree(None,\
                             HuffmanTree(None, HuffmanTree(5), HuffmanTree(6)),\
                             HuffmanTree(15)))
    >>> t == result
    True
    >>> freq == {5: 5, 6: 6, 7: 7, 8: 8, 15: 15}
    True

    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    >>> freq == {symbol: 6}
    True
    """
    if not freq_dict:
        return HuffmanTree(None)
    if len(freq_dict) == 1:
        dummy_byte = (list(freq_dict.keys())[0] + 1) % 256
        dummy_freq = freq_dict.copy()
        dummy_freq[dummy_byte] = 0
        q = Queue(dummy_freq)
    else:
        q = Queue(freq_dict)
    while q.size > 1:
        (tree1, freq1) = q.dequeue()
        (tree2, freq2) = q.dequeue()
        new_tree = HuffmanTree(None, tree1, tree2)
        new_freq = freq1 + freq2
        q.enqueue(new_tree, new_freq)
    return q.dequeue()[0]


######
# below is the helper function for creating H-trees.

class Queue:
    """This queue is to store the order that a tree got in the H-trees.
    It is the sorted queue according to freq_dict.

    Public Attributes:
    ===========
    size: the number of left trees in the queue
    duilie: the list to store the queue putting in the H-tree.
    freq: the dictionary to store the frequency of the byte in the trees.

    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> q = Queue(freq)
    >>> duilie = [HuffmanTree(3), HuffmanTree(7), HuffmanTree(2)]
    >>> q.duilie == duilie
    True
    >>> q.size
    3
    >>> q.freq
    [4, 5, 6]
    """
    size: int
    duilie: List[int]
    freq: List[int]  # the index is the same in self.duilie. It is dangerous,

    # but have to since H-trees cannot represent as a key.

    def __init__(self, freq_dict: Dict[int, int]) -> None:
        """Creating the queue from the freq_dict.
        Anything plug in after this should NOT be the tree representing a byte.
        """
        f = sorted(freq_dict.items(), key=lambda x: x[1])
        self.size = len(f)
        self.duilie = []
        self.freq = []
        for i in f:
            self.duilie.append(HuffmanTree(i[0]))
            self.freq.append(i[1])

    def enqueue(self, tree: HuffmanTree, tree_freq: int) -> None:
        """Add a new tree into the queue, self.duilie, in order.

        precondition: self.duilie is sorted according to self.freq.
                      tree has not been recorded in self.freq yet, which means
        this tree is NOT representing a byte.
        """
        if self.size == 0:
            self.duilie = [tree]
            self.freq = [tree_freq]
            self.size += 1
            return

        for i in range(len(self.duilie)):
            if self.freq[i] > tree_freq:
                self.duilie = self.duilie[:i] + [tree] + self.duilie[i:]
                self.freq = self.freq[:i] + [tree_freq] + self.freq[i:]
                self.size += 1
                return

        self.duilie += [tree]
        self.freq += [tree_freq]
        self.size += 1

    def dequeue(self) -> (HuffmanTree, int):
        """Return the next one should get in the H-tree and its frequency, and
        delete them in the queue.
        """
        self.size -= 1
        tree = self.duilie.pop(0)
        freq = self.freq.pop(0)
        return tree, freq


# upper is the helper function for creating H-trees.
######

def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree == HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    True

    >>> freq = {11: 7, 14: 4, 12: 5, 13: 2}
    >>> tree = build_huffman_tree(freq)
    >>> d = get_codes(tree)
    >>> d == {11: "0", 12: "10", 13: "110", 14: "111"}
    True
    >>> tree == build_huffman_tree(freq)
    True

    >>> freq = {1: 6}
    >>> tree = build_huffman_tree(freq)
    >>> d = get_codes(tree)
    >>> d[1] == "1"
    True
    >>> tree == build_huffman_tree(freq)
    True

    >>> tree = HuffmanTree()
    >>> d = get_codes(tree)
    >>> not d
    True
    """
    if tree == HuffmanTree(None):
        return {}
    if tree.is_leaf():
        return {tree.symbol: ''}
    else:
        left = get_codes(tree.left)
        for key in left:
            left[key] = "0" + left[key]
        right = get_codes(tree.right)
        for key in right:
            right[key] = "1" + right[key]
        right.update(left)
        return right


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> tree == HuffmanTree(None, left, right)
    True


    >>> freq = {5: 5, 6: 6, 7: 7, 8: 8, 15: 15}
    >>> tree = build_huffman_tree(freq)
    >>> number_nodes(tree)
    >>> tree.number
    3
    >>> tree.right.number
    2
    >>> tree.right.right.number
    >>> tree.right.left.number
    1
    >>> tree.left.number
    0
    >>> tree == build_huffman_tree(freq)
    True
    """
    if tree == HuffmanTree(None):
        return
    number_helper(tree, 0)


def number_helper(tree: HuffmanTree, number: int) -> int:
    """Number the internal nodes of the tree. Return the next number to take.
    """
    if not tree.is_leaf():
        if tree.left.is_leaf() and tree.right.is_leaf():
            tree.number = number
            return number + 1
        left = number_helper(tree.left, number)
        right = number_helper(tree.right, left)
        tree.number = right
        return right + 1
    return number


#####
# This is the helper function for numbering.
class Stack:
    """ADT Stack to store the order to number.

    Public Attributes:
    ===========
    storage: the stack.
    """
    storage: List

    def __init__(self) -> None:
        """Make a empty storage of the stack
        """
        self.storage = []

    def is_empty(self) -> bool:
        """
        >>> s = Stack()
        >>> s.is_empty()
        True
        >>> s.push('hello')
        >>> s.is_empty()
        False
        """
        return not self.storage

    def push(self, item: any) -> None:
        """Push a item.
        """
        self.storage.append(item)

    def pop(self) -> any:
        """
        >>> s = Stack()
        >>> s.push('hello')
        >>> s.push('goodbye')
        >>> s.pop()
        'goodbye'
        """
        return self.storage.pop()


#####

def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    >>> tree == HuffmanTree(None, left, right)
    True

    >>> freq = {}
    >>> avg_length(tree, freq)
    0.0
    """
    if tree == HuffmanTree(None) or not freq_dict:
        return 0.0
    dic = {}
    s = 0.0
    n = 0
    get_weight(tree, 0, dic)
    for i in freq_dict:
        s += dic[i] * freq_dict[i]
        n += freq_dict[i]
    return s / n


def get_weight(tree: HuffmanTree, weight: int, dic: Dict) -> None:
    """Get the weight for every symbol in tree. Store in dic.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> dic = {}
    >>> get_weight(tree, 0, dic)
    >>> dic == {3: 2, 2: 2, 9: 1}
    True
    """
    if tree.is_leaf():
        dic[tree.symbol] = weight
    else:
        get_weight(tree.left, weight + 1, dic)
        get_weight(tree.right, weight + 1, dic)


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text == bytes([1, 2, 1, 0])
    True

    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> text == bytes([1, 2, 1, 0, 2])
    True

    >>> text = bytes()
    >>> result = compress_bytes(text, d)
    >>> result == b''
    True
    >>> text == bytes()
    True

    >>> text = bytes([1, 1, 1, 1])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10101010']
    >>> text == bytes([1, 1, 1, 1])
    True
    """
    # source: https://www.runoob.com/python3/python3-func-bytes.html
    # source: http://c.biancheng.net/view/2175.html
    if not text:
        return bytes()
    lst = [codes[i] for i in text]
    bit = ''.join(lst)
    result = bytes([bits_to_byte(bit[i: i + 8]) \
                    for i in range(0, len(bit), 8)])
    return result


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> tree == HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    True


    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree == HuffmanTree(None, left, right)
    True


    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> tree == build_huffman_tree(build_frequency_dict(b"helloworld"))
    True

    >>> tree = HuffmanTree()
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    []
    >>> tree == HuffmanTree()
    True

    >>> freq = {11: 7, 14: 4, 12: 5, 13: 2}
    >>> tree = build_huffman_tree(freq)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 13, 0, 14, 0, 12, 1, 0, 0, 11, 1, 1]
    """
    if tree == HuffmanTree(None):
        return bytes()
    if tree.is_leaf():
        return bytes()
    else:
        byt = bytes()
        if not tree.left.is_leaf():
            byt += tree_to_bytes(tree.left)
        if not tree.right.is_leaf():
            byt += tree_to_bytes(tree.right)
        if tree.left.is_leaf():
            byt += bytes([0, tree.left.symbol])
        else:
            byt += bytes([1, tree.left.number])
        if tree.right.is_leaf():
            byt += bytes([0, tree.right.symbol])
        else:
            byt += bytes([1, tree.right.number])
    return byt


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst
    [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 1, 1, 0)]
    """
    if not node_lst:
        return HuffmanTree()
    root_node = node_lst[root_index]
    if root_node.l_type == 0:
        left = HuffmanTree(root_node.l_data)
    else:
        left = generate_tree_general(node_lst, root_node.l_data)
    if root_node.r_type == 0:
        right = HuffmanTree(root_node.r_data)
    else:
        right = generate_tree_general(node_lst, root_node.r_data)
    return HuffmanTree(None, left, right)


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> lst
    [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 0, 1, 0)]

    >>> freq = {11: 7, 14: 4, 12: 5, 13: 2}
    >>> tree = build_huffman_tree(freq)
    >>> number_nodes(tree)
    >>> lst = bytes_to_nodes(tree_to_bytes(tree))
    >>> generate_tree_postorder(lst, 999) == tree
    True
    >>> lst
    [ReadNode(0, 13, 0, 14), ReadNode(0, 12, 1, 0), ReadNode(0, 11, 1, 1)]

    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> lst = bytes_to_nodes(tree_to_bytes(tree))
    >>> generate_tree_postorder(lst, 999) == tree
    True
    >>> lst
    [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114), ReadNode(1, 0, 1, 1), \
ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, 3), ReadNode(1, 2, 1, 4)]
    """
    if not node_lst:
        return HuffmanTree()
    i = root_index * 0
    stack = Stack()
    while i < len(node_lst):
        if node_lst[i].r_type == 0:
            right = HuffmanTree(node_lst[i].r_data)
        else:
            right = stack.pop()
        if node_lst[i].l_type == 0:
            left = HuffmanTree(node_lst[i].l_data)
        else:
            left = stack.pop()
        stack.push(HuffmanTree(None, left, right))
        i += 1
    return stack.pop()


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'

    >>> decompress_bytes(HuffmanTree(), b'', 100)
    b''

    >>> decompress_bytes(tree, b'', 0)
    b''
    """
    if size == 0 or tree == HuffmanTree():
        return b''
    dic = get_bytes(tree, '')
    t = ''.join([byte_to_bits(j) for j in text])
    lst = []
    curr_byt = ""
    for i in t:
        curr_byt += i
        if curr_byt in dic:
            lst.append(dic[curr_byt])
            curr_byt = ""
    return bytes(lst)[:size]


def get_bytes(tree: HuffmanTree, code: str) -> Dict[str, int]:
    """Return the dictionary which maps the codes of every symbols according to
    the tree.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_bytes(tree , '')
    >>> d == {"0": 3, "1": 2}
    True
    >>> tree == HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    True

    >>> freq = {11: 7, 14: 4, 12: 5, 13: 2}
    >>> tree = build_huffman_tree(freq)
    >>> d = get_bytes(tree, '')
    >>> d == {"0": 11, "10": 12, "110": 13, "111": 14}
    True
    >>> tree == build_huffman_tree(freq)
    True

    >>> freq = {1: 6}
    >>> tree = build_huffman_tree(freq)
    >>> d = get_bytes(tree, '')
    >>> d["1"] == 1
    True
    >>> tree == build_huffman_tree(freq)
    True

    >>> tree = HuffmanTree()
    >>> d = get_bytes(tree, '')
    >>> not d
    True
    """
    if tree == HuffmanTree(None):
        return {}
    if tree.is_leaf():
        return {code: tree.symbol}
    else:
        left = get_bytes(tree.left, code + "0")
        right = get_bytes(tree.right, code + "1")
        right.update(left)
        return right


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    lst = sorted(get_depth(tree, 0), key=lambda x: x[1], reverse=True)
    if len(lst) <= 1:
        return
    for i in range(len(lst) - 1):
        for j in range(len(lst) - i - 1):
            leaf1 = lst[j][0]
            leaf2 = lst[j + 1][0]
            if freq_dict[leaf1.symbol] > freq_dict[leaf2.symbol]:
                leaf1.symbol, leaf2.symbol = leaf2.symbol, leaf1.symbol


def get_depth(tree: HuffmanTree, depth: int) -> List[(HuffmanTree, int)]:
    """Get the depth for every symbol in tree.
    RETURN LIST IS NOT SORTED!!!

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> get_depth(tree, 0)
    [(HuffmanTree(3, None, None), 2), (HuffmanTree(2, None, None), 2), \
(HuffmanTree(9, None, None), 1)]
    """
    if tree.is_leaf() or tree == HuffmanTree(None):
        return [(tree, depth)]
    else:
        lst = []
        lst += get_depth(tree.left, depth + 1)
        lst += get_depth(tree.right, depth + 1)
        return lst


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
