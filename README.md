# Action Sets: Weakly Supervised Action Segmentation without Ordering Constraints
Code for the paper Action Sets: Weakly Supervised Action Segmentation without Ordering Constraints

### Prepraration:

* download the data from https://uni-bonn.sciebo.de/s/wOxTiWe5kfeY4Vd
* extract it so that you have the `data` folder in the same directory as `main.py`
* create a  `results` directory in the same directory where you also find `main.py`: `mkdir results`

Requirements: Python3.x with the libraries numpy, pytorch (version 0.4.1), and scipy

### Training:

Run `python main.py training`

### Inference:

Run `python main.py inference --n_threads=NUM_THREADS`, where `NUM_THREADS` should be replaced with the number of parallel CPU threads you want to use for Viterbi decoding.

### Evaluation:

In the inference step, recognition files are written to the `results` directory. The frame-level ground truth is available in `data/groundTruth`. Run `python eval.py --recog_dir=results --ground_truth_dir=data/groundTruth` to evaluate the frame accuracy of the trained model

### Remarks:

We provide a python/pytorch implementation for easy usage. In the paper, we used a faster, in-house C++ implementation, so results can be slightly different. Running the provided setup on split1 of Breakfast should lead to roughly 23% frame accuracy.

If you use the code, please cite

    A. Richard, H. Kuehne, J. Gall:
    Action Sets: Weakly Supervised Action Segmentation without Ordering Constraints
    in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2018
