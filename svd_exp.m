#!/usr/bin/octave

## make a matrix representing a cube
cube = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
## visualize the cube
plot3(cube(:,1), cube(:,2), cube(:,3), "+");
## perform svd on the cube [cube = u*v*w']
[u, v, w] = svd(cube);

## translate the cube
cube2 = cube + 1;
plot3(cube2(:,1), cube2(:,2), cube2(:,3), "x");
[u2, v2, w2] = svd(cube2);

##
cube3 = u*v*(w + 1)';
plot3(cube3(:,1), cube3(:,2), cube3(:,3), "*");

## sparse the cube and repeat above steps
