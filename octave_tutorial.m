% Elementary and conditional operations
% Changing the prompt:
PS('>> ');

%%%% matrices %%%%
A = [1 2; 3 4; 5 6]

v = [1 3 4]
v = [1; 3; 4]
v = 1:0.1:2
ones(2,3) % generate a matrix of 1's
zeros(1,3) % ditto
rand(3,3) % random
eye(6) % identity

help rand % help with any command

%%%% how to move data around %%%%

size(A) % [3 2] three rows and 2 columns

% load featuresX.data
% save hello.mat v: saves v into a new file hello.mat

% Operation with matrices
A = [1 2; 3 4; 5 6]
B = [8 9; 10 11; 12 13]

A * B
A .* B % . is the element wise operator
% element-wise operations log, abs, exp, max this last retuns the column-wise max

A' % transpose
t = [0:0.001:1]
y1 = sin(2.*pi*4*t)
y2 = cos(2.*pi*4*t)

%
plot(t, y2)
plot(t, y1)
hold on;
plot(t, y2, 'r')
xlabel('time')
ylabel('function')
legend('sin', 'cos')
title('my plot')
print -dpng 'myPlot.png'

figure(2); plot(t, y2)

subplot(1,2,1) % divides the plot area into 1 row 2 columns and holds the first one
plot(t, y1) % plots it on the previously holded area
subplot(1,2,2)
plot(t, y2)
imagesc(A)
imagesc(A), colorbar, colormap gray;

%%%%% Loops
for i=1:10
  v(i) = 2^i;
end;

while i <= 5
  v(i) = 100
  i = i+1
end;

while true
  v(i) = 999
  i = i + 1

  if i == 6
   break
  end
end

%% vectorized routines

% itetrative (not so bueno)
prediction  = 0
for j = 1:n+1
  prediction = prediction + theta(j) * x(j)
end;

% vectorized
prediction = theta' * x
% this is better cuz it's using highly optimized algorithm provided by octave
