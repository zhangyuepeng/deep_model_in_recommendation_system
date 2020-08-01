# Multi-interest network with dynamic routing for recommendation at Tmall


## Example to run the codes.
The params is integrated in the code. and, the code is about only the train stage.

```
python MIND.py
```

## Dataset
I provide two processed datasets: MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20). 

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

<br/>
<br/>

**refer to [https://github.com/hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)**
**refer to [https://github.com/shenweichen/DeepMatch](https://github.com/shenweichen/DeepMatch)**
