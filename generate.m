clear;
clc;
close all;
X = [];
I = 1;
A = [.7 -.1 0 0;
    .2 -.5 .1 0;
    0 .1 .1 0
    .5 0 .5 .5];
B = [0 .1;
    .1 1;
    .1 0;
    0 0];
Q = eye(4);
R = eye(2);
N = 10;
[~, S, ~] = dlqr(A, B, Q, R);

n = 10000;
rand('seed', 2000);
x0 = rand(4, n);
x0(1:2, :) = x0(1:2, :)*12 - 6;
x0(3, :) = x0(3, :)*2 - 1;
x0(4, :) = x0(4, :) - .5;
U = [];

for a = 1:n
    cvx_begin quiet
        cvx_solver SDPT3
        variable x(4, N)
        variable u(2, N-1)
        s = 0;
        for b = 1 : N-1
            s = s+x(:, b)'*Q*x(:, b)+u(:, b)'*R*u(:, b);
        end
        s = s+x(:, N)'*S*x(:, N);
        minimize(s)
        subject to
            abs(x(1:2, :)) <= 6
            abs(x(3, :)) <= 1
            abs(x(4, :)) <= .5
            abs(u) <= 5
            x(:, 1) == x0(:, a)
            for b = 1 : N-1
                x(:, b+1) == A*x(:, b) + B*u(:, b)
            end
    cvx_end
    if strcmp(cvx_status, 'Solved')
        X = [X, x0(:, a)];
        U = [U, u(:)];
    else
        cvx_begin quiet
            cvx_solver SeDuMi
            variable x(4, N)
            variable u(2, N-1)
            s = 0;
            for b = 1 : N-1
                s = s+x(:, b)'*Q*x(:, b)+u(:, b)'*R*u(:, b);
            end
            s = s+x(:, N)'*S*x(:, N);
            minimize(s)
            subject to
                abs(x(1:2, :)) <= 6
                abs(x(3, :)) <= 1
                abs(x(4, :)) <= .5
                abs(u) <= 5
                x(:, 1) == x0(:, a)
                for b = 1 : N-1
                    x(:, b+1) == A*x(:, b) + B*u(:, b)
                end
        cvx_end
        if strcmp(cvx_status, 'Solved')
            X = [X, x0(:, a)];
            U = [U, u(:)];
        end
    end
    disp([num2str(a) '/' num2str(n)]);
end
save data.mat X U