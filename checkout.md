**DATAFAWN**



**Current checkout**

* entirely rewrite Zeni
* need to keep track of the animal's direction
* probably will need to keep a rolling max\_length for each leg (to account for camera zoom / animal getting closer)
* criteria:
    1. Foot reaches lowest vertical position (detect peaks in y\_pos)
    2. foot is moving downward. I think it should be the frame before y\_vel hit zero, but check 
    3. forward velocity decreases sharply (negative acceleration).I think it should be once acceleration flips to negative

