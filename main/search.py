# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from builtins import object
import util


class SearchProblem(object):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST

    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    # Initialize stack, visited list, and list of actions
    stack = util.Stack()
    visited = []

    # Add start node to stack
    startNode = (problem.getStartState(), [])
    stack.push(startNode)

    # Begin search
    while stack.isEmpty() == False:

        # Grab current node and current path
        currNode = stack.pop()

        # If goal is found, search is complete
        if problem.isGoalState(currNode[0]):
            return currNode[1]

        # Check if new node
        nodeVisited = False

        for node in visited:
            if currNode[0] == node:
                nodeVisited = True
                break

        # If new node, search. Otherwise, move to next node in queue
        if nodeVisited == True:
            continue

        else:
            visited += [currNode[0]]
            
            # Get successors of node
            successors = problem.getSuccessors(currNode[0])

            # Check successors and add nodes + actions to stack
            for node in successors:

                # Update actions to include node
                actions = currNode[1] + [node[1]]

                # Push new position and set of actions
                newNode = (node[0], actions)
                stack.push(newNode)

    return currNode[1]

# DFS algorithm only with queue instead of stack
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Initialize queue, visited list, and list of actions
    queue = util.Queue()
    visited = []

    # Add start node to queue
    startNode = (problem.getStartState(), [])
    queue.push(startNode)

    # Begin search
    while queue.isEmpty() == False:

        # Grab current node and current path
        currNode = queue.pop()

        # If goal is found, search is complete
        if problem.isGoalState(currNode[0]):
            return currNode[1]

        # Check if new node
        nodeVisited = False

        for node in visited:
            if currNode[0] == node:
                nodeVisited = True
                break

        # If new node, search. Otherwise, move to next node in queue
        if nodeVisited == True:
            continue

        else:
            visited += [currNode[0]]
            
            # Get successors of node
            successors = problem.getSuccessors(currNode[0])

            # Check successors and add nodes + actions to queue
            for node in successors:
                
                # Update actions to include node
                actions = currNode[1] + [node[1]]

                # Push new position and set of actions
                newNode = (node[0], actions)
                queue.push(newNode)

    return currNode[1]

# Based on DFS/BFS algorithm but now adding in basic cost calculation
def uniformCostSearch(problem, heuristic=None):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Initialize queue, visited list, and list of actions
    queue = util.PriorityQueue()
    visited = []

    # Add start node to queue
    startNode = (problem.getStartState(), [])
    queue.push(startNode, 0)

    # Begin search
    while queue.isEmpty() == False:

        # Grab current node and current path
        currNode = queue.pop()

        # If goal is found, search is complete
        if problem.isGoalState(currNode[0]):
            return currNode[1]

        # Check if new node
        nodeVisited = False

        for node in visited:
            if currNode[0] == node:
                nodeVisited = True
                break

        # If new node, search. Otherwise, move to next node in queue
        if nodeVisited == True:
            continue

        else:
            visited += [currNode[0]]
            
            # Get successors of node
            successors = problem.getSuccessors(currNode[0])

            # Check successors and add nodes + actions to queue
            for node in successors:
                
                # Update actions and cost to include node
                actions = currNode[1] + [node[1]]
                cost = problem.getCostOfActions(actions) # Get cost for priority in queue, unlike DFS and BFS

                # Push new position and set of actions
                newNode = (node[0], actions)
                queue.push(newNode, cost)

    return currNode[1]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# UniformCostSearch but includes heuristic calculation in cost for priority
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize queue, visited list, and list of actions
    queue = util.PriorityQueue()
    visited = []

    # Add start node to queue
    startNode = (problem.getStartState(), [], 0) # Add cost to node for searching
    queue.push(startNode, 0)

    # Begin search
    while queue.isEmpty() == False:

        # Grab current node and current path
        currNode = queue.pop()

        # Go ahead and add to visited for check later
        visited += [(currNode[0], currNode[2])]

        # If goal is found, search is complete
        if problem.isGoalState(currNode[0]):
            return currNode[1]

        # Get successors of node
        successors = problem.getSuccessors(currNode[0])

        # Check successors and add nodes + actions to queue
        for node in successors:

            # Update actions and cost to include node
            actions = currNode[1] + [node[1]]
            cost = problem.getCostOfActions(actions) # Get cost for priority in queue, unlike DFS and BFS
            priority = cost + heuristic(node[0], problem) # Calculate priority based on cost and heuristic

            # Check node isn't already visited with lower cost
            higherCost = False
            for visitNode in visited:
                if visitNode[0] == node[0]:
                    if visitNode[1] <= cost:
                        higherCost = True
                        break

            # If new or lower cost, push new position and set of actions
            if higherCost == False:
                visited += [(node[0], cost)]
                newNode = (node[0], actions, cost)
                queue.push(newNode, priority)

    return currNode[1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
