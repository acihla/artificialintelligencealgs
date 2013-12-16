# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
		#print successorGameState.getScore()
		#print newPos
		#print newFood
		#print newGhostStates
		#print newScaredTimes

		"*** YOUR CODE HERE ***"
		newGhostPos = successorGameState.getGhostPosition(1)
		newGameScore = successorGameState.getScore()
		newDistToGhost = manhattanDistance(newPos, newGhostPos)

		food = newFood
		listFoodDistance = []
		for i in range(0, food.width):
			for j in range(0, food.height):
				if food[i][j] == True:
					#listFoodDistance.append(abs(position[0] - i) + abs(position[1] - j))
					listFoodDistance.append(((i,j), (manhattanDistance(newPos, (i,j)))))
		closestFood = newPos
		minDist = food.width * food.height
		for food in listFoodDistance:
			if food[1] < minDist:
				closestFood = food[0]
				minDist = food[1]

		foodGrid = newFood
		listFoodDistance = [0]
		sumDist = 0.0
		counter = 0.0
		for i in range(0, foodGrid.width):
			for j in range(0, foodGrid.height):
				if foodGrid[i][j] == True:
					posToConsider = (i,j)
					distBetween = manhattanDistance(newPos, posToConsider)
					sumDist += distBetween
					counter += 1
		if counter > 1:
			foodDensity = (sumDist/counter)
		else:
			foodDensity = 1
		

		if newPos == newGhostPos:
		  return newGameScore - abs(newGameScore)
		#if newPos == closestFood:
		#  return newGameScore + abs(newGameScore)
		evalFuncScore = newGameScore - minDist / foodDensity + newDistToGhost / foodDensity
		#print evalFuncScore
		return evalFuncScore

def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""


	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game
		"""
		"*** YOUR CODE HERE ***"
		
		numAgents = gameState.getNumAgents()
		depth = self.depth
		calcs = numAgents * depth
		pacmanActions = gameState.getLegalActions(0)
		currentState = gameState

		#print pacmanActions
		pacmanSuccessors = []
		for action in pacmanActions:
			pacmanSuccessors.append(currentState.generateSuccessor(0,action))
		#print pacmanSuccessors
		listOfMaxes = []
		for successor in pacmanSuccessors:
			listOfMaxes.append(self.MinMaxHelper(numAgents, calcs - 1, 1, successor))
		#print listOfMaxes
		maxIndex = 0
		maxTies = []
		definiteMax = max(listOfMaxes)
		for i in range(0,len(listOfMaxes)):
			if (listOfMaxes[maxIndex] < listOfMaxes[i]):
				maxIndex = i
			if (definiteMax == listOfMaxes[i]):
				maxTies.append(listOfMaxes[i])
		"""
		if len(maxTies) > 1:
			print "WE HIT A TIE MOTHERFUCKER"
			return Directions.STOP
		"""

		return pacmanActions[maxIndex]


	def MinMaxHelper(self, numAgents, calcs, agent, state):
		agentActions = state.getLegalActions(agent)

		if (calcs == 0 or state.isLose() or state.isWin()):
			return self.evaluationFunction(state)

		agentSuccessors = []
		for action in agentActions:
			agentSuccessors.append(state.generateSuccessor(agent, action))


		if agent == 0:
			pacmanEvalScores = []
			for successor in agentSuccessors:
				pacmanEvalScores.append(self.MinMaxHelper(numAgents, calcs - 1, (agent + 1) % numAgents, successor))
			return max(pacmanEvalScores)

		else:
			agentEvalScores = []
			for successor in agentSuccessors:
				agentEvalScores.append(self.MinMaxHelper(numAgents, calcs - 1, (agent + 1) % numAgents, successor))
			return min(agentEvalScores)
	


		"""
		while depth > 0:
			agentCount = numAgents - 1
			while agentCount >= 0:
				possibleActions = currentState.getLegalActions(agentCount)
				minMaxScoreList = []
				for action in possibleActions:
					successorAgent = currentState.generateSuccessor(agentCount, action)
					minMaxScoreList.append((action, successorAgent))
				if agentCount == 0:
					agentPickState = max(minMaxScoreList, key=lambda item: item[1])
				else:
					agentPickState = min(minMaxScoreList, key=lambda item: item[1])
				#print agentPickState
				currentState = agentPickState[1]
				agentCount -= 1
			depth -= 1
		return agentPickState[0]
		
		num
		util.raiseNotDefined()"""
		

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""

		actions = gameState.getLegalActions( 0 )
		agentTotal = gameState.getNumAgents()
		
		bestScore = -(sys.maxint)
		bestAction = Directions.STOP
		alpha = -(sys.maxint)
		beta = sys.maxint

		for action in actions:
			successor = gameState.generateSuccessor( 0, action )
			actionScore = self.ABHelper( successor, 1, self.depth, alpha, beta )
			alpha = max( alpha, actionScore )
			if ( actionScore > bestScore ):
				bestScore = actionScore
				bestAction = action
		
		return bestAction

	def ABHelper(self, state, agent, depth, alpha, beta ):
		
		agentTotal = state.getNumAgents()
		if ( agent == agentTotal ):
			depth -= 1
			agent = 0
		
		if ( state.isLose() or state.isWin() or depth == 0 ):
			return self.evaluationFunction(state)
		elif ( agent == 0 ):
			v = -(sys.maxint)
			actions = state.getLegalActions( agent )
			for action in actions:
				successor = state.generateSuccessor( agent, action )
				v = max( v, self.ABHelper( successor, agent+1, depth, alpha, beta ) )
				if ( v > beta ):
					return v
				alpha = max( alpha, v )
			return v
		else:
			v = sys.maxint
			actions = state.getLegalActions( agent )
			for action in actions:
				successor = state.generateSuccessor( agent, action )
				v = min( v, self.ABHelper( successor, agent+1, depth, alpha, beta ) )
				if ( v < alpha ):
					return v
				beta = min( beta, v )
			return v

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		numAgents = gameState.getNumAgents()
		depth = self.depth
		calcs = numAgents * depth
		pacmanActions = gameState.getLegalActions(0)
		currentState = gameState

		#print pacmanActions
		pacmanSuccessors = []
		for action in pacmanActions:
			pacmanSuccessors.append(currentState.generateSuccessor(0,action))
		#print pacmanSuccessors
		listOfMaxes = []
		for successor in pacmanSuccessors:
			listOfMaxes.append(self.ExpectiMaxHelper(numAgents, calcs - 1, 1, successor))
		#print listOfMaxes
		maxIndex = 0
		maxTies = []
		definiteMax = max(listOfMaxes)
		for i in range(0,len(listOfMaxes)):
			if (listOfMaxes[maxIndex] < listOfMaxes[i]):
				maxIndex = i
			if (definiteMax == listOfMaxes[i]):
				maxTies.append(listOfMaxes[i])
		"""
		if len(maxTies) > 1:
			print "WE HIT A TIE MOTHERFUCKER"
			return Directions.STOP
		"""

		return pacmanActions[maxIndex]


	def ExpectiMaxHelper(self, numAgents, calcs, agent, state):
		agentActions = state.getLegalActions(agent)

		if (calcs == 0 or state.isLose() or state.isWin()):
			return self.evaluationFunction(state)

		agentSuccessors = []
		for action in agentActions:
			agentSuccessors.append(state.generateSuccessor(agent, action))


		if agent == 0:
			pacmanEvalScores = []
			for successor in agentSuccessors:
				pacmanEvalScores.append(self.ExpectiMaxHelper(numAgents, calcs - 1, (agent + 1) % numAgents, successor))
			return max(pacmanEvalScores)

		else:
			agentEvalScores = []
			for successor in agentSuccessors:
				agentEvalScores.append(self.ExpectiMaxHelper(numAgents, calcs - 1, (agent + 1) % numAgents, successor))
			return sum(agentEvalScores)/len(agentEvalScores)
	

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	  So there was quite a bit of guess-and-checking happening when dealing with the capsule/ghost scenario
	  However, for the most part I recycled quite a bit from my Q1 response and added in the capsule/ghost
	  factor which seemed to help alot. Basically my Q1 checked for food density and weighted states 
	  as more valuable when they were surrounded by more food (really it was just minimizing the average
		distance to food... not true density). 

		I was seriously lucky when I guessed 1.2 (after guessing 2) as an appropriate multiplier for my capsule/
		chasing scared ghost scenario. I later checked 1.1 and 1.3 which performed substantially worse.
	"""
	"*** YOUR CODE HERE ***"

	newPos = currentGameState.getPacmanPosition()
	newFood = currentGameState.getFood()
	newGhostStates = currentGameState.getGhostStates()
	newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

	newGhostPos = currentGameState.getGhostPosition(1)
	newGameScore = currentGameState.getScore()
	newDistToGhost = manhattanDistance(newPos, newGhostPos)
	food = newFood
	listFoodDistance = []
	for i in range(0, food.width):
		for j in range(0, food.height):
			if food[i][j] == True:
				#listFoodDistance.append(abs(position[0] - i) + abs(position[1] - j))
				listFoodDistance.append(((i,j), (manhattanDistance(newPos, (i,j)))))
	closestFood = newPos
	minDist = food.width * food.height
	for food in listFoodDistance:
		if food[1] < minDist:
			closestFood = food[0]
			minDist = food[1]

	foodGrid = newFood
	listFoodDistance = [0]
	sumDist = 0.0
	counter = 0.0
	for i in range(0, foodGrid.width):
		for j in range(0, foodGrid.height):
			if foodGrid[i][j] == True:
				posToConsider = (i,j)
				distBetween = manhattanDistance(newPos, posToConsider)
				sumDist += distBetween
				counter += 1
	if counter > 1:
		foodDensity = (sumDist/counter)
	else:
		foodDensity = 1
		
	capsuleGrid = currentGameState.getCapsules()
	#print capsuleGrid
	#print newPos
	multiplier = 1
	for capsulePos in capsuleGrid:
		if manhattanDistance(newPos, capsulePos) < 4 and newDistToGhost < 4:
			#print "this is a capsule" 
			multiplier = 1.2
	if newPos in capsuleGrid:
			print "this is a capsule" 

	if newPos == newGhostPos:
		return newGameScore - abs(newGameScore)
		#if newPos == closestFood:
		#  return newGameScore + abs(newGameScore)
	evalFuncScore = multiplier * newGameScore - minDist / foodDensity + newDistToGhost / foodDensity
		#print evalFuncScore
	
	
	return evalFuncScore

# Abbreviation
better = betterEvaluationFunction

def manhattanHeuristic(position, problem, info={}):
	"The Manhattan distance heuristic for a PositionSearchProblem"
	xy1 = position
	xy2 = problem.goal
	return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def foodHeuristicValue(state):
	position = state.getPacmanPosition()
	foodGrid = state.getFood()
	listFoodDistance = []
	sumDist = 0.0
	counter = 0.0
	for i in range(0, foodGrid.width):
		for j in range(0, foodGrid.height):
			if foodGrid[i][j] == True:
				#posToConsider = (i,j)
				#listFoodDistance.append(abs(position[0] - i) + abs(position[1] - j))
				distToFood = abs(position[0] - i) + abs(position[1] - j)
				if (distToFood < min(foodGrid.width, foodGrid.height)):
					sumDist += distToFood
					counter += 1
	tiebreaker = 1
	#print foodGrid
	#print foodGrid[position[0]][position[1]]
	if position in foodGrid:
		print "we are hitting food"
		tiebreaker = 2

	retVal = (sumDist/counter) * tiebreaker
	if counter > 0:
		print retVal
		return retVal
	else:
		return 0


class ContestAgent(MultiAgentSearchAgent):
	"""
	  Your agent for the mini-contest
	"""

	def getAction(self, gameState):
		"""
		  Returns an action.  You can use any method you want and search to any depth you want.
		  Just remember that the mini-contest is timed, so you have to trade off speed and computation.

		  Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
		  just make a beeline straight towards Pacman (or away from him if they're scared!)
		"""
		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()

