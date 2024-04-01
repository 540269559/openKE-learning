# 这三行代码都是为了让python2兼容python3
# 1.当前模块中所有的导入都必须使用绝对导入方式:
# 绝对导入是指，当你导入一个模块时，Python会首先搜索当前目录，然后是Python路径。这是为了避免与标准库中的模块同名的模块产生冲突。
from __future__ import absolute_import
# 除法"/"默认为浮点数除法:兼容python2
from __future__ import division
# 让print可以作为函数在python2中
from __future__ import print_function

from .Trainer import Trainer
from .Tester import Tester

"""
__all__ 是一个特殊的python变量
在模块中定义了 __all__ 变量时，它指定了在使用 from module import * 语句时导入的符号
在下面这个代码中,__all__列表含了 Trainer 和 Tester，
因此当其他模块使用 from 当前模块 import * 语句时，只会导入 Trainer 和 Tester 这两个符号。
综上所述:其他文件引入这个文件的时候,只会导出Trainer和Tester这两个文件
"""
__all__ = [
    'Trainer',
    'Tester'
]
