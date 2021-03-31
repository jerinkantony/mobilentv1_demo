import multiprocessing
from time import time

from multiprocessing import Pool
import os
import numpy as np
import random
from time import sleep


def parallel_proc_folder(subfolderfunc, src_root, dst_root, other_args):  
    '''calls subfolderfunc on each subfolder'''
    count=0
    result_objs=[]
    with Pool(processes=os.cpu_count()-1) as pool:
        for folderpath, subdirs, srcfiles in os.walk(src_root):
            count+=1
            result = pool.apply_async(subfolderfunc, args=(folderpath,srcfiles,src_root,dst_root,count,other_args))
            print('File count:',count)
            result_objs.append(result)
        results = [result.get() for result in result_objs]
        print('sample_parallelop', len(results))    
    print('parallel_proc Done', count)
    return results


if 1:
    
    def process_file_func1(fpath,dstpath):
        '''actual processing function, else it can be added to process_subfolder func directly'''
        pass
    
    def process_subfolder(subfolderpath,srcfiles,src_root,dst_root,count,other_args):
        ''' loop through folder here'''
        arg1,arg2,arg3=other_args['arg1'],other_args['arg2'],other_args['arg3']
        print(count,arg1,arg2,arg3)
        for f in srcfiles:
            fp = os.path.join(subfolderpath, f)
            process_file_func1(fp,dstpath='somepath')
            '''else implement func here'''
            pass  
            
        #sleepTime = random.uniform(0, 0.1)
        #print( " requires ", sleepTime, " seconds to finish")
        #print('count', count, subfolderpath, len(srcfiles))
        #sleep(sleepTime)
        
        return count #return any stuff here
    
    root = '/home/skycam/siamese/speaker_siamese/DB/siamese_net_data'
    #pattern = "*.wav"
    other_args = dict(arg1='hi',arg2=1,arg3='there')
    
    resutls = parallel_proc_folder(subfolderfunc = process_subfolder, src_root=root, dst_root='somepath',other_args=other_args)
    print('Results from main',len(resutls))






'''
import multiprocessing
from itertools import permutations

values = [1, 2, 3, 4, 5]
l = permutations(values, 2)
data=[]

def f(x):
    return x[0], x[1], x[0] + x[1]

with multiprocessing.Pool(5) as p:
    data = p.map(f, l)

print(data)


from multiprocessing import Pool
from time import sleep
import random



from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if 1:
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print (q.get() )   # prints "[42, None, 'hello']"
    p.join()
    

from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if 1:
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()
  
  
#Apply async      
def sum(task, a, b):
    sleepTime = random.randint(1, 2)
    print(task, " requires ", sleepTime, " seconds to finish")
    sleep(sleepTime)
    return a+b


def printResult(result):
    print(result)

if 0:
    myPool = Pool(2)
    res = []
    for i in range(5):
        myPool.apply_async(sum, args=("task1", i, i+1,), callback = printResult)

    print("Submitted tasks to pool")
    myPool.close()
    myPool.join()
    print('res',res)
    


from multiprocessing import Pool
import os
import numpy as np

def f(n):
    return np.var(np.random.sample((n, n)))


def sample_parallelop():
    result_objs = []
    n = 10

    with Pool(processes=os.cpu_count() - 1) as pool:
        for _ in range(n):
            result = pool.apply_async(f, (n,))
            result_objs.append(result)
        
        results = [result.get() for result in result_objs]
        print('sample_parallelop', len(results), np.mean(results), np.var(results))
        
    return results
   
if 1:
    sample_parallelop()    
'''

'''
import os
from fnmatch import fnmatch

root = '/home/skycam/siamese/speaker_siamese/DB/siamese_net_data'
#pattern = "*.wav"
count=0
for path, subdirs, files in os.walk(root):

    for name in files:
        count+=1
        #print (os.path.join(path, name))
        #if fnmatch(name, pattern):
        #    print os.path.join(path, name)
        
print('File count:',count)
'''













         
            
            
            

'''
def square(x):
    # calculate the square of the value of x
    return x*x

if __name__ == '__main__':

    # Define the dataset
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Output the dataset
    print ('Dataset: ' + str(dataset))

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    agents = 5
    chunksize = 1
    with Pool(processes=agents) as pool:
        result = pool.map(square, dataset, chunksize)
        
    # Output the result
    print ('Result:  ' + str(result))
    
    with Pool(5) as pool:
        res = pool.map(square, dataset)
    
    print('res    :',res)
 '''
 

'''
from functools import partial

def target(lock, a, b, item): 
    # Do cool stuff
    if (True):
        lock.acquire()
        print(a,b,item)
        lock.release()
        
def example1():
    a='hi'
    b='there'
    iterable = [1, 2, 3, 4, 5]
    
    pool = multiprocessing.Pool()
    m = multiprocessing.Manager()
    l = m.Lock()
    func = partial(target, l, a, b)
    pool.map(func, iterable)
    pool.close()
    pool.join()

if __name__ == '__main__':
    example1()
'''


''' 
import multiprocessing
import os
import time


def worker_main(queue):
    print (os.getpid(),"working")
    while True:
        item = queue.get(True)
        print( os.getpid(), "got", item)
        time.sleep(1) # simulate a "long" operation

def example2():
    the_queue = multiprocessing.Queue()
    the_pool = multiprocessing.Pool(3, worker_main,(the_queue,))
    #                            don't forget the coma here  ^

    for i in range(5):
        the_queue.put("hello")
        the_queue.put("world")

    time.sleep(10)

if __name__ == '__main__':
    #example1()
    example2()
'''

'''
#from functools import partial

def func(c):
    lock.acquire()
    print("{} {} {}".format(a, b, c))
    lock.release()

def init_pool(l,a_,b_):
    global lock, a, b
    lock,a,b = l,a_,b_   
    
def main():
    iterable = [1, 2, 3, 4, 5]
    
    a = "hi"
    b = "there"
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init_pool, initargs=(l, a, b))
    
    #func = partial(func,l, a, b)
    pool.map(func, iterable)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main() 
'''
'''
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

import numpy as np


# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
data[:5]


def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

tic = time()
results = []
for row in data:
    results.append(howmany_within_range(row, minimum=4, maximum=8))
print('time=',time()-tic)
print(results[:10])

#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]


#apply
# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
pool.close()    

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]


# Parallelizing using Pool.map()
import multiprocessing as mp

# Redefine, with only 1 mandatory argument.
def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count
tic = time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(howmany_within_range_rowonly, [row for row in data])
pool.close()
print('time2=', time()-tic)

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]


# Parallelizing with Pool.starmap()

tic = time()
pool = mp.Pool(mp.cpu_count())
results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])

pool.close()
print('time2=', time()-tic)

print(results[:10])

'''


      
    
    
    
    
