# Cellural Automata

## Types of cellular automatons
- Homogeneous cellural automata (HCA)

    >The ruleset of the automata only considers the number of neighbours of a certain type of states but not their positions
- Non-Homogeneous cellural automata (NHCA)

	>The ruleset works with the positions of the states in the neigbourhood

## Moore Neighbourhood based cellural automatas
-	[Wikipedia](https://en.wikipedia.org/wiki/Moore_neighborhood)

> **Note**: The rest of the study consideres _HCA_-s until said differently  

### Efficiency study
**n** := width of the simulated area  
**m** := height of the simulated area  
**k** := size of the considered neighbourhood (k = 3 in Moore Neighbourhoods) 

The simulated area is considered to be round wrapping

#### Generating next Matrix from previous
| Operation         | _Accurances_ | _Time overhead_ |
|-------------------|--------------|-----------------|
| _iteration_ 	    | 1     	   | n * m    	     |
| neighbour queries | n * m   	   | k * k    	     |
| applying ruleset  | tbd   	   | tbd     	     |

**Performace considerations:**

- Iteration

	>If the Matrix is stored as a 2d array it's slightly inefficient
	We can do a flat `n * m` array and when advancing in the standard row major order then 
	we only increment the index by one, hence avoiding a multiply-add(in C with compile time 2d array) or a double dereference in other pointer in pointer representations, in this case the cache has much less work to do

- Neighbour queries

	>A standard implementation for neigboor querry (in HCA) needs to return the number of cells in the neighbourhood for every needed state
	The needed states depend on the ruleset, but for simplicity from now it should return the number of cells in the neigbourhood for _every_ state
	In practice it could be a HashSet: State -> Int 

- Applying the ruleset

	>Not yet explored


#### Restricting the general model for better performance
**Restrictions:**

- We will only consider Moore Neighbourhood type HCA
- We want to store the 3 * 3 neigbourhood of states as a 128 SSE registrer
	- We let 
- The states are integers from 0 to LAST_STATE 
- LAST_STATE &le 255;

**Implications:**

- We can make a HashSet: State -> Int as an array of size 255
- The states fit in a byte

#### Arhitecture
Lets name the result as `Board` represented as an n \* m matrix

Lets focus only on the cells not the edge, since the edge points and internal points compare as follows: O(n + m) << O(n\*m) Meaning the internal points are orders of magnutide larger, so we shoul d optimize those and later come back to fix the points on the edge of the simulated area

We store the simulated area as a flat array in this order:
```
 (1)  (2)  (3) | (19) (20) (21) 
 (4)       (5) | (22)      (23)
 (6)  (7)  (8) | (24) (25) (26)
 (9) (10) (11) | (27)      (27)
(12)      (13) | (28) (29) (30)
(14) (15) (16) | (31)      (32)
 ```
Lets call this `MatrixN11` meaning a matrix for the neigbours of (1, 1) as a subcell

In this representation when we load an index of 5 * i 64 bit wide then we get the neigbours of the "middle point" but it isn't that easy with the other point's neightbours

PER_SLICE := 3m

Here (0 based) from 5i to 5i + 7 we have the NW NN NE WW EE SW SS SE neighbours in this order of 
(0)  -> (1, 1)  (18) -> (4, 1)
(5)  -> (1, 3)  (23) -> (4, 3)
(10) -> (1, 5)  (28) -> (4, 5)
Valid Points: 
	(k): PER_SLICE * c + 5 * b
		Where
			c = floor(t / PER_SLICE)
			b = t mod PER_SLICE  

So a number k is mapped by the above formula to the 8 neigbours of the point (1 + c, 1 + 2 * b)
(k) -> (1 + c, 1 + 2 * b)
But only when k is in the form PER_SLICE * c + 5 * b
Which is every 6 th element (considering only the points inside the simulated area)

We could Just store another 5 matrixes like this, buts it is inefficient, lets look for another solution

Can I transform this Matrix into any other?
If we shift the flat array representation of the array by 3 then we get `MatrixN12`
	For the middle points it is trivial, and for the points 
