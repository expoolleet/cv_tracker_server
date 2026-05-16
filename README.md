# Embedded video streaming server with object-tracking capabilities
This server-based system was developed using libraries like sockets, OpenCV, FFTW, Numpy to track arbitrary objects using single-board computers.

In particular, it was tested on a Raspberry Pi 3B+ and Raspberry Pi Zero 2 W.

Results of testing various algorithms on a Raspberry Pi 3B+:
* Own custom tracker produces 80 FPS avg. with 73.99% accuracy.
* MOSSE	produces 110 FPS avg. with 44.50% accuracy.
* KCF	produces 51 FPS avg. with	57.55% accuracy.
* CSRT produces	18 FPS avg. with 79.82% accuracy.
