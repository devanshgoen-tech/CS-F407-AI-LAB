% Optional Prolog extension (logic_lab_ex.pdf, Section 7).
%
% Facts describing the warehouse connections, together with the
% can_move/2 and valid_move/2 rules used to independently verify
% movements proposed by the Python planner.
%
% Example queries (SWI-Prolog):
%   ?- can_move(a, b).      % true
%   ?- can_move(a, c).      % false (no direct edge; see path/2 for reachability)
%   ?- valid_move(b, c).    % true
%   ?- valid_move(a, c).    % false
%   ?- path(a, c, P).       % P = [a, b, c] .

connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

can_move(X, Y) :- connected(X, Y).

valid_move(X, Y) :- connected(X, Y).

% Reachability (Task 7 Challenge): is Move(a, c) supported?
% Direct answer: no, but there is a two-step path.
path(X, X, [X]).
path(X, Y, [X|Rest]) :-
    connected(X, Z),
    path(Z, Y, Rest),
    \+ member(X, Rest).

% Task 8: a tiny rule chain to show forward inference.
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
