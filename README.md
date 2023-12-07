# inception_i3d
Pytorch implementation of the Inception I3d model proposed by Carreira and Zisserman

The code is super ugly. Will try to clean it soon.  
The outputs of both models are not 100% the same of some reason. I'll investigate.

With RGB only, ImageNet pretrained, top predictions:

Pytorch:
```
Class: 227, output: 1.0, logits: 126.66184997558594
Class: 153, output: 1.656426046437366e-21, logits: 78.81222534179688
Class: 48, output: 1.5140779010039856e-22, logits: 76.41978454589844
Class: 50, output: 4.321102704888711e-26, logits: 68.25814819335938
Class: 237, output: 8.502026767294067e-30, logits: 59.72460174560547
Class: 256, output: 7.33277725372041e-32, logits: 54.97148132324219
Class: 150, output: 5.61040884155942e-32, logits: 54.70375061035156
Class: 168, output: 1.5313968468753322e-33, logits: 51.102725982666016
Class: 161, output: 3.6517460952545115e-35, logits: 47.366573333740234
Class: 86, output: 2.3472268901332033e-35, logits: 46.92461013793945
```

Tensorflow:
```
Class: 227, output: 0.99999666214, logits: 25.8566532135
Class: 237, output: 1.33534240376e-06, logits: 12.3303337097
Class: 48, output: 4.55312829217e-07, logits: 11.2543754578
Class: 297, output: 3.14340724117e-07, logits: 10.8838682175
Class: 50, output: 1.92432480617e-07, logits: 10.3931360245
Class: 358, output: 1.30964608047e-07, logits: 10.0083179474
Class: 166, output: 1.06817452661e-07, logits: 9.80451202393
Class: 143, output: 9.44640632383e-08, logits: 9.68161010742
Class: 168, output: 7.84288545219e-08, logits: 9.49558258057
Class: 153, output: 7.80173650128e-08, logits: 9.49032211304
```
