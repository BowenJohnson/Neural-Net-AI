// Bowen Johnson
// Summer 2021


using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetExample
{
    class Program
    {
        static void Main(string[] args)
        {
            int rowSize = 8;
            int colSize = 8;

            // create the net
            Net theNet = new Net();

            // instantiate the neurons
            Neuron[,] input = new Neuron[rowSize, colSize];
            Neuron outX = new Neuron();
            Neuron outO = new Neuron();
            theNet.AddOutput(outX);
            theNet.AddOutput(outO);

            for (int idx = 0; idx < 8; idx++)
            {
                for (int jdx = 0; jdx < 8; jdx++)
                {
                    input[idx, jdx] = new Neuron();
                    theNet.AddInput(input[idx, jdx]);
                    theNet.Connect(input[idx, jdx], outX);
                    theNet.Connect(input[idx, jdx], outO);
                }
            }

            // load training data
            List<TrainingData> data = new List<TrainingData>();
            string trainingFileDir = "TrainingFiles";
            DirectoryInfo trainingDir = new DirectoryInfo(trainingFileDir);

            foreach (FileInfo TrainingFile in trainingDir.GetFiles("*.txt"))
            {
                // read file
                string fileContents = File.ReadAllText(TrainingFile.FullName);

                // put files into training data
                string[] lines = fileContents.Split('\n');

                double[] expectedOut = null;
                if (lines[0][0] == 'X')
                {
                    // output is {1.0, 0.0}
                    expectedOut = new double[] { 1.0, 0.0 };
                }
                else if (lines[0][0] == 'O')
                {
                    // output is {0.0, 1.0}
                    expectedOut = new double[] { 0.0, 1.0 };
                }

                // process inputs
                double[] inputVal = new double[64];
                int index = 0;

                for (int idx = 1; idx <= 8; idx++)
                {
                    for (int jdx = 0; jdx <= 21; jdx += 3)
                    {
                        inputVal[index] = lines[idx][jdx] == '1' ? 1.0 : 0.0;
                        index++;
                    }
                }

                // make training point in the list
                data.Add(new TrainingData(inputVal, expectedOut));
            }

            // train the net
            theNet.Train(data.ToArray());

            // write the output and weights
            string outFile = "";
            for (int idx = 0; idx < 8; idx++)
            {
                for (int jdx = 0; jdx < 8; jdx++)
                {
                    outFile += theNet.GetSynapse(input[idx, jdx], outX).Weight.ToString("0.0000");
                    outFile += ", ";
                    outFile += theNet.GetSynapse(input[idx, jdx], outO).Weight.ToString("0.0000");

                    if (jdx < 7)
                    {
                        outFile += ", ";
                    }
                }
                if (idx < 7)
                {
                    outFile += "\n";
                }
            }

            File.WriteAllText("out.txt", outFile);

            // load and eval test data
            // print if it's an X or O
            string testFileDirName = "TestFiles";
            DirectoryInfo testDir = new DirectoryInfo(testFileDirName);

            foreach (FileInfo testFile in testDir.GetFiles("*.txt"))
            {
                // read in the file
                string fileCont = File.ReadAllText(testFile.FullName);

                // set values
                string[] lines = fileCont.Split('\n');

                // process input
                Console.WriteLine("Input:");

                for (int idx = 0; idx < 8; idx++)
                {
                    for (int jdx = 0; jdx < 8; jdx++)
                    {
                        input[idx, jdx].Value = lines[idx + 1][jdx * 3] == '1' ? 1.0 : 0.0;
                        Console.Write(lines[idx + 1][jdx * 3]);
                    }

                    Console.WriteLine();
                }

                // determine output
                theNet.Evaluate();

                // print results to screen
                Console.WriteLine("X = " + outX.Value.ToString());
                Console.WriteLine("O = " + outO.Value.ToString());
            }
        }
    }

    class Net
    {
        public Net()
        {
            inputs = new List<Neuron>();
            outputs = new List<Neuron>();
            synapses = new List<Synapse>();
        }

        private List<Neuron> inputs;
        private List<Neuron> outputs;
        private List<Synapse> synapses;

        public void AddInput(Neuron n) { inputs.Add(n); }
        public void AddOutput(Neuron n) { outputs.Add(n); }

        public void Connect(Neuron from, Neuron to, double weight = 0.0)
        {
            Synapse s = new Synapse();
            s.Axon = from;
            s.Dentrite = to;
            s.Weight = weight;
            synapses.Add(s);
        }

        public Synapse GetSynapse(Neuron from, Neuron to)
        {
            foreach (Synapse s in synapses)
            {
                if (s.Axon == from && s.Dentrite == to)
                    return s;
            }
            return null;
        }

        public void Evaluate()
        {
            foreach (Neuron outNeuron in outputs)
            {
                double value = 0.0;
                foreach (Neuron inNeuron in inputs)
                {
                    Synapse s = GetSynapse(inNeuron, outNeuron);
                    value += s.Weight * inNeuron.Value;
                }
                outNeuron.Value = value;
            }
        }

        public void Train(TrainingData[] data)
        {
            // train the net using gradient descent

            // set weights to random values
            Random r = new Random();
            foreach (Synapse s in synapses)
            {
                s.Weight = r.NextDouble() * 2 - 1.0;  // value between -1.0 and 1.0
            }

            // minimize the error
            double learningRate = 0.01;
            double precision = 0.01;
            double lastError;
            double currentError = double.MaxValue;
            do
            {
                lastError = currentError;
                currentError = 0.0;
                foreach (Synapse s in synapses)
                    s.dW = 0.0;

                // for each training point...
                foreach (TrainingData d in data)
                {
                    // for each output neuron...
                    for (int j = 0; j < outputs.Count; j++)
                    {
                        // calculate Yj from inputs and weights
                        outputs[j].Value = 0.0;
                        for (int i = 0; i < inputs.Count; i++)
                        {
                            Synapse s = GetSynapse(inputs[i], outputs[j]);
                            outputs[j].Value += s.Weight * d.X[i];
                        }

                        // determine error contribution from this output node and training point
                        currentError += Math.Pow(d.T[j] - outputs[j].Value, 2.0);

                        // determine weight gradient for each synapse
                        for (int i = 0; i < inputs.Count; i++)
                        {
                            Synapse s = GetSynapse(inputs[i], outputs[j]);
                            s.dW += (d.T[j] - outputs[j].Value) * d.X[i];
                        }
                    }
                }

                // update error for number of training points
                currentError /= data.Length;

                // adjust weights
                foreach (Synapse s in synapses)
                    s.Weight += learningRate * s.dW;
            }
            while (Math.Abs(currentError - lastError) > precision);
        }
    }

    class Neuron
    {
        public double Value { get; set; }
    }

    class Synapse
    {
        public Neuron Axon { get; set; }        // output
        public Neuron Dentrite { get; set; }    // input
        public double Weight { get; set; }
        public double dW { get; set; }          // for training only
    }

    public class TrainingData
    {
        public TrainingData()
        {
            X = new List<double>();
            T = new List<double>();
        }

        public TrainingData(double[] input, double[] expected)
        {
            X = new List<double>(input);
            T = new List<double>(expected);
        }

        public List<double> X { get; set; }     // input values
        public List<double> T { get; set; }     // expected output values
    }
}
