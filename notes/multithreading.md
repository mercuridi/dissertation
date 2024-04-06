# Multithreading
## Modularities analysis
Current estimate now that everything is working, 60 hours to process the data.
Currently working:
- Loop that actually processes everything
- Sentiment calculation
- Toxicity calculation
- Modularities included
- Collating calculated data into final csv
- Writing final csv
Currently, it takes about 7 minutes to process about 2000 entries. That means, for 1m~ entries, we're looking at 60 hours (1,000,000/2,000)*7/60
Will look into multithreading python for quicker loads and a better spread on the gpu.

Multithreading sucks; multiprocessing might be good
https://docs.python.org/3/library/multiprocessing.html
Would need to do:
- Locking on print statements to prevent garbled output
- Include global counters as multiprocessing Value objects
    - Maybe just remove them? They don't add a massive amount when we consider that we're already using timers.
    - If I want to count the value objects, I can just bring the logic back to a single thread and use disslib to load the files and just count them
- Rework main for loop body as a function to be called by a pool, probably using the forkserver start method
    - Function should take the lists of files to process as argument
    - Note the example:
```
if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```
    - So the eventual multithreaded implementation will likely have "f" as the body INSIDE the current for loop, and the list of values as pairs of filenames

done, and doesn't help because the multiprocessing fills memory and causes a crash
