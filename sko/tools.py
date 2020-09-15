import numpy as np

"该工具函数用于将demo测试函数转化维想要的函数类型" \
"demo的函数格式和想要的函数格式说明如下"
def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible
    :param func:
    :return:
    '''

    prefered_function_format = '''
        def demo_func(x):
             x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    '''

    is_vector = getattr(func, 'is_vector', False)
    if is_vector:
        return func
    else:
        # print(func.__code__)
        # 返回已经编译的函数对象，这里就是demo自定义的函数
        # <code object demo_func at 0x000001FC30102D20, file "E:/SouthChinaUni/群智能算法学习/scikit-opt-master/examples/demo_pso.py", line 6>
        if func.__code__.co_argcount == 1:
            def func_transformed(X):
                # print(np.array([func(x) for x in X]))
                return np.array([func(x) for x in X])
            # print(func_transformed)
            return func_transformed
        elif func.__code__.co_argcount > 1:

            def func_transformed(X):
                return np.array([func(*tuple(x)) for x in X])
            # print(func_transformed)
            return func_transformed

    raise ValueError('''
    object function error,
    function should be like this:
    ''' + prefered_function_format)
