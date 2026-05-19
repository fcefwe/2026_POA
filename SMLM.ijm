// ImageJ macro: detect blobs in channel2 and keep only those pixels

selectWindow("channel2");
run("Duplicate...", "title=BlobMask duplicate");
selectWindow("BlobMask");
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast...", "saturated=0.35 normalize process_all");
run("8-bit");

// Small blur
run("Duplicate...", "title=g1 duplicate");
selectWindow("g1");
run("Gaussian Blur...", "sigma=1 stack");

// Large blur
selectWindow("BlobMask");
run("Duplicate...", "title=g2 duplicate");
selectWindow("g2");
run("Gaussian Blur...", "sigma=4 stack");

// DoG
imageCalculator("Subtract create stack", "g1", "g2");
selectWindow("Result of g1");
rename("DoG");
run("Enhance Contrast", "saturated=0.35");

// Threshold and mask
setAutoThreshold("MaxEntropy dark");

imageCalculator("Multiply create 32-bit stack", "BlobMask","DoG");
selectImage("Result of BlobMask");

run("Mean...", "radius=1 stack");

setAutoThreshold("MaxEntropy dark");
setOption("BlackBackground", true);
run("Convert to Mask", "method=MaxEntropy background=Dark calculate black");

close("g1");
close("g2");