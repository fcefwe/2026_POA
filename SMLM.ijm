// ImageJ macro: detect blobs in the currently active image and keep only those pixels
// Saves input1 and input2 automatically in the SAME folder as the source image.

// Capture the currently active source image dynamically.
title = getTitle();
selectWindow(title);

// Get the folder containing the source image.
// getDirectory("image") already returns the directory where the image was loaded from.
outputDir = getDirectory("image");

if (outputDir == "")
    exit("Could not determine the folder of the currently active image.");

if (!endsWith(outputDir, File.separator))
    outputDir = outputDir + File.separator;


// ============================================================
// Create input1 from the currently active source image
// ============================================================

selectWindow(title);

run("Duplicate...", "title=input1 duplicate");

selectWindow("input1");

run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast...", "saturated=0.35 normalize process_all");

run("8-bit");


// ============================================================
// Small Gaussian blur
// ============================================================

run("Duplicate...", "title=g1 duplicate");

selectWindow("g1");

run("Gaussian Blur...", "sigma=1 stack");


// ============================================================
// Large Gaussian blur
// ============================================================

selectWindow("input1");

run("Duplicate...", "title=g2 duplicate");

selectWindow("g2");

run("Gaussian Blur...", "sigma=4 stack");


// ============================================================
// Difference of Gaussians
// ============================================================

imageCalculator("Subtract create stack", "g1", "g2");

selectWindow("Result of g1");

rename("DoG");

run("Enhance Contrast", "saturated=0.35");


// ============================================================
// Threshold DoG
// ============================================================

setAutoThreshold("MaxEntropy dark");


// ============================================================
// Multiply original processed image by DoG
// ============================================================

imageCalculator("Multiply create 32-bit stack", "input1", "DoG");

selectWindow("Result of input1");

rename("input2");


// ============================================================
// Process input2
// ============================================================

run("Mean...", "radius=1 stack");

setAutoThreshold("MaxEntropy dark");

setOption("BlackBackground", true);

run("Convert to Mask", "method=MaxEntropy background=Dark calculate black");

run("Mean...", "radius=2 stack");


// ============================================================
// Save input1 and input2
// ============================================================

input1Path = outputDir + "input1.tif";
input2Path = outputDir + "input2.tif";


// Delete old files if they already exist,
// preventing ImageJ from asking for confirmation.

if (File.exists(input1Path))
    File.delete(input1Path);

if (File.exists(input2Path))
    File.delete(input2Path);


// Save input1.

selectWindow("input1");

saveAs("Tiff", input1Path);


// Save input2.

selectWindow("input2");

saveAs("Tiff", input2Path);


// ============================================================
// Clean up temporary images
// ============================================================

selectWindow("g1");
close();

selectWindow("g2");
close();

selectWindow("DoG");
close();