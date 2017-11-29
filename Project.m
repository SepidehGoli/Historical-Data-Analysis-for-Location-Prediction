function [ p , v] = Project( p, v, a)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if (nargin == 2)
    p = p + v;
end
if (nargin == 3)
    p = p + 0.5 * a + v;
    v = v + a;
end

