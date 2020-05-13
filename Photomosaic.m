clear all

Mural = imageDatastore({'IMG_1.JPG','IMG_2.JPG', 'IMG_3.JPG', 'IMG_4.JPG', 'IMG_5.JPG'})
I = readimage(Mural,1);
grayImage = rgb2gray(I);
intrinsics = cameraIntrinsics([3527.13139, 3496.68293], [2061.33721, 1523.30019], [480, 640]);

[J,newOrigin] = undistortImage(grayImage,intrinsics);
intrinsics;

[y,x,m] = harris(grayImage,10000,'tile',[.1 .1]);
points = [x,y];
[features, points] = extractFeatures(grayImage, points);
%imshow(G,'Border','tight')

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
numImages = numel(Mural.Files);
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);
imageSize(1,:) = size(grayImage);
% Iterate over remaining image pairs
position(1,1) = 0;
for n = 2:numImages
    
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = readimage(Mural, n);

    % Convert image to grayscale.
    grayImage = rgb2gray(I);

    % Save image size.
    imageSize(n,:) = size(grayImage);

    % Detect and extract SURF features for I(n).
    [y,x,m] = harris(grayImage,10000,'tile',[.1 .1])
    points = [x,y];
    [features, points] = extractFeatures(grayImage, points);

    
    C = (imageSize(1,:)/2);
    %xi = C(1);
    %yi = C(2);
    % Find correspondences between I(n) and I(n-1).
    [indexPairs,matchmetric] = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    shift(1,n) = max(matchedPoints(2,:))
    position(n,1) = .5*imageSize(n-1,1) - (imageSize(n,1) - shift(1,n)) + .5*imageSize(n,1) + position(n-1,1)
    
    shift(2,n) = mean(matchedPoints(1,:) -(matchedPointsPrev(1,:)))
    position(n,2) = shift(2,n) + 320;
    
    %x(i) = min(matchedPoints(1,1)) - xi;
    %y(i) = min(matchedPoints(1,2)) - yi;
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
    'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T;
end

% Compute the output limits  for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3],'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask','MaskSource', 'Input port');
%blender = vision.AlphaBlender('Operation', 'Blend','MaskSource','Input port')

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);
text =  cell(5,1)
value = zeros(length(shift),2)
% Create the panorama.
ix = 1000
iy = 350
for i = 1:numImages

    I = readimage(Mural, i);
     
    text{i} = ['center image: ' num2str(i) '\n Position: ' num2str(position(i,1)) ',' num2str(position(i,2))];
    
    value(1,1) = position(1,1); 
    value(1,2) = position(1,1);
    
    value(2,1) = position(2,2);
    value(2,2) = position(2,1);
    
    value(3,1) = position(3,2); 
    value(3,2) = position(3,1);
    
    value(4,1) = position(4,2); 
    value(4,2) = position(4,1);
    
    value(5,1) = position(5,2); 
    value(5,2) = position(5,1);
    
    value(6,1) = position(6,2); 
    value(6,2) = position(6,1);
    
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage,mask);
    %panorama = [panorama,warpedImage];
    
end
%panorama = imresize(panorama,[600 450]);

RGB = insertText(panorama,value,text,'FontSize',18);
%panorama = imresize(panorama,[450 600]);
%panorama = imresize(panorama,[240 890]);
panorama = imresize(panorama, .5);
figure
imshow(panorama)
imshow(RGB)
