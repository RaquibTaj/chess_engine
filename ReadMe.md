# Chess game
### Credits: [George Hotz](https://www.twitch.tv/georgehotz)
### The plan:-

Create a Zero Knowledge Chess Engine.  
Use a neural network to prune the search tree.  

### Mathematical Defination:
```
V = f(board)
```
V: a value of -1 represents the black win state, a value of 0 represents a draw, and a value of 1 represents white win state.
<br>  
All positions where white wins = 1  
All positions where draw = 0  
All positions where black wins = -1

### State (Board):
```
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
```

### Legal Moves representation:
```
g1h3
g1f3
b1c3
b1a3
h2h3
g2g3
f2f3
e2e3
d2d3
c2c3
b2b3
a2a3
h2h4
g2g4
f2f4
e2e4
d2d4
c2c4
b2b4
a2a4
```
We represnt the legal moves as edges over a graph.

### Pieces:
1. Blank (no piece exist)
2. Pawn
3. Rook
4. Rook (can castle)
5. Knight
6. Bishop
7. Queen
8. King
9. Blank, with possible en passant

### Extra States:
1. To move


### Data Structure:
```
(8x8x4) + 1 = 257 bits
```
257 bits stored in a vector of 0's or 1's. The "+1" is used to represnt whose move is next. 

### Training Dataset:
To train the model, we use the free KingBase 2019 database, consisting of around 2.2 million chess games played since 1990 with a player ELO rating greater than 2000
