Tasks:

Acquire training data (rip lots of frames from some anime/animation) [Richard]
Get edge detection working [Roger Liu]
Get optical flow working w.r.t the edge detected images [Roger Liu]
Create network architecture by borrowing from: [Sohil]
	pix2pix: https://affinelayer.com/pixsrv/
	Research papers on interpolation: 
		https://esc.fnwi.uva.nl/thesis/centraal/files/f1305544686.pdf
		http://cs229.stanford.edu/proj2016/report/KorenMendaSharma-ConvolutionalNeuralNetworkForVideoFrameCompression-report.pdf


Installing OpenCV:
install the appropriate wheel here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
	I used opencv_python-3.2.0+contrib-cp35-cp35m-win_amd64.whl because I have 64-bit architecture and CPython 3.5

Basic structure (advice given by Zico):

Take x frames, predict inbetween frame

Look at generative adversarial net architecture:
  Framework can produce "realistic" looking images
  Look it up: optical flow on the edges
  
Picture -> edge dected -> optical flow on the edge map -> fill in the image with generative adversarial net

Data set:
Look at what pix2pix does


Test time:
2 images + edges + middle frame edges -> full image

Train time:
take in 2 images + edges + optical flow interpolation -> produces full image				



Look at waifu2xd for cleaning up 

