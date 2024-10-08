﻿using System;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

class MazeSolver
{
    public static  List<string> CheckList = new List<string>();

    public static void CheckPositionHistory((int row, int col) position, string direction)
    {
        var myString = $"{position.row}, {position.col}, {direction}";

        if (!CheckList.Any(x => x == myString))
        {
            // Add the entry to the list if it doesn't already exist
            CheckList.Add(myString);
            Console.WriteLine(myString);
        }
    }

    static void Main(string[] args)
    {
        // Define the maze as a 2D numpy array (0 - path, 1 - wall)
        int[,] mazeArr = new int[,]
        {
            { 0, 1, 0, 0, 0, 0 },
            { 0, 1, 1, 1, 1, 0 },
            { 0, 0, 1, 0, 0, 0 },
            { 1, 0, 1, 0, 1, 0 },
            { 0, 0, 0, 0, 0, 0 }
        };

        var maze = np.array(mazeArr);

        // Define the start and goal positions
        var start = (row: 0, col: 0);
        var goal = (row: 0, col: 2);

        // Define the actions (up, down, left, right)
        var actions = new[] { "up", "down", "left", "right" };
        int nActions = actions.Length;

        // Define the Q-table as a 3D Tensor (rows, columns, action space)
        var qTable = tf.Variable(tf.zeros(new int[] { Convert.ToInt32(maze.shape[0]), Convert.ToInt32(maze.shape[1]), nActions }));

        // Parameters
        float learningRate = 0.1f;
        float discountFactor = 0.9f;
        float epsilon = 0.1f; // Exploration rate
        int episodes = 1000;

        // Helper functions
        (int row, int col) GetNextPosition((int row, int col) position, string action)
        {
            int row = position.row, col = position.col;
            switch (action)
            {
                case "up":
                    {
                        return (Math.Max(row - 1, 0), col);
                    }
                case "down":
                    {
                        return(Math.Min(Convert.ToInt32(row + 1), Convert.ToInt32(maze.shape[0] - 1)), col);
                    }
                case "left":
                    {
                        return (row, Math.Max(col - 1, 0));
                    }
                case "right":
                    {
                        return (row, Convert.ToInt32(Math.Min(col + 1, maze.shape[1] - 1)));
                    }
                default: return position;
            }
        }

        int GetReward((int row, int col) position)
        {
            if (position == goal)
            {
                // Reward for reaching the goal

                //Console.WriteLine($"Goal Reached at Position: {position.row}, {position.col}");

                return 10;
            }
            if (maze[position.row, position.col] == 1)
            {
                // Penalty for hitting a wall

                //Console.WriteLine($"Position in x, y: {position.row}, {position.col}");

                return -1;
            }

            // Neutral reward
            return 0;
        }

        // Training the Q-table
        for (int episode = 0; episode < episodes; episode++)
        {
            var position = start;
            while (position != goal)
            {
                // Choose an action
                int actionIdx;
                if (np.random.randn() < epsilon)
                {
                    actionIdx = np.random.randint(nActions);
                }
                else
                {
                    var qValues = qTable[position.row, position.col];
                    actionIdx = np.argmax(qValues.numpy());
                }

                var action = actions[actionIdx];
                var nextPosition = GetNextPosition(position, action);

                CheckPositionHistory(position, action);

                // Get the reward for the next position
                int reward = GetReward(nextPosition);

                // Update the Q-value
                var qValuesNext = qTable[nextPosition.row, nextPosition.col];
                float bestNextAction = tf.reduce_max(qValuesNext).numpy(); // Using tf.reduce_max instead of np.max
                var target = reward + discountFactor * bestNextAction;

                var qValue = qTable[position.row, position.col].numpy()[actionIdx];
                qValue = (1 - learningRate) * qValue + learningRate * target;
                var newQValues = qTable[position.row, position.col].numpy();
                newQValues[actionIdx] = qValue;
                qTable[position.row, position.col].assign(tf.constant(newQValues));

                // Move to the next position
                if (maze[nextPosition.row, nextPosition.col] == 0)
                {
                    position = nextPosition;
                }
            }
        }

        // Extract the optimal path
        var optimalPath = new List<(int row, int col)>();
        var currentPosition = start; // Changed from 'position' to 'currentPosition'
        optimalPath.Add(currentPosition);
        while (currentPosition != goal)
        {
            var qValues = qTable[currentPosition.row, currentPosition.col];
            int actionIdx = np.argmax(qValues.numpy());
            var action = actions[actionIdx];
            currentPosition = GetNextPosition(currentPosition, action);
            optimalPath.Add(currentPosition);
        }

        Console.WriteLine("Optimal path found by Q-learning:");
        foreach (var pos in optimalPath)
        {
            Console.WriteLine($"({pos.row}, {pos.col})");
        }

        Console.WriteLine("\n\n Original Maze\n\n");

        PrintMaze(mazeArr);

        Console.WriteLine("\n\n");

        PrintMazeOptimalPath(mazeArr, optimalPath);

        Console.ReadLine();

    }

    public static void PrintMaze(int[,] maze)
    {
        int rows = maze.GetLength(0);
        int cols = maze.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Console.Write(maze[i, j] + " ");
            }
            Console.WriteLine(); // Move to the next line after printing each row
        }
    }

    public static void PrintMazeOptimalPath(int[,] maze, List<(int row, int col)> optPath)
    {
        int rows = maze.GetLength(0);
        int cols = maze.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Check if the current position is part of the optimal path
                if (optPath.Contains((i, j)))
                {
                    Console.Write("* ");
                }
                else
                {
                    Console.Write(maze[i, j] + " ");
                }
            }
            Console.WriteLine(); // Move to the next line after printing each row
        }
    }
}
