// the plan is to create a basic, regressional mlp which approximates a function.
// the constraint is that i'm not allowed to use ANY external libraries
// hi github
// i wrote comments just for you

class Layer {
    constructor(parameters, nonLinearity) {

        // the last parameter will be the bias => [w_1, w_2, ..., w_n-1, w_n, b] per neuron
        // it would look like [neuron, neuron, neuron], sp [[w, b], [w, b], ...] then flattened to [w, b, w, b, ...]
        // non linear functions will be handled by the gpu (webgpu) so we need to read some wgsl or buffer or something, im not sure how

        this.parameters = parameters;
        this.nonLinearity = nonLinearity;

    }
    forward(input) {

        // right now in a cpu based system, input refers to the array passed by the previous layer, but later it should refer to buffer location

        const inputDimension = input.length;
        const quotient = this.parameters.length / (inputDimension + 1); // how many indexes, if its an integer

        // and then, assuming that parameters and input size cleanly pro
        if (
            Number.isInteger(quotient)
        ) {
            // for all the neurons, fill them with 0 (placeholder) then map that to specific weights, bias; (thank god js doesnt need type identifiers)
            const neuronParameters = new Array(quotient).fill(0).map((_, i) => {
                const sectionStart = (inputDimension + 1) * i;
                return this.parameters.slice(sectionStart, sectionStart + inputDimension + 1);
            });

            let output = [];

            for (const parameters of neuronParameters) { // for every neuron in parameters
                let dotProduct = parameters[parameters.length - 1]; // set it to bias first so that 
                for (let i = 0; i < parameters.length - 1; i++) {
                    dotProduct = dotProduct + (parameters[i] * input[i]);
                }

                output.push(dotProduct);
            }

            return output;
        }

    }
    backward() {

    }
}

class NeuralNetwork {
    constructor(layers) { // ouu shii i forgot that getting device n adapter is async so i move to init method
        this.device = null;
        this.layers = layers;

        // layers and stuff go here
    }

    // BUT, i wont touch GPU right now, too little time

    async init() { // initialise webgpu stff
        const gpu = navigator.gpu;
        if (gpu) { //icl im just using an if statement and nesting everything in there bc it looks cooler

            const adapter = await gpu.requestAdapter();
            this.device = await adapter.requestDevice();

            // then i set the batches and stuff and the pipeline.... later. not here; plus, each layer should have its own wgsl definition

        } else {
            console.error("no gpu found")
        }
    }
}

// and so running should look something like

new NeuralNetwork([
    new Layer(),
    new Layer(),
    new Layer(),
]);

// const input; //whatever input would be (user selected)
// const [l1, l2, l3] = [new Layer(), new Layer(), new Layer()];

// // compositional function form
// l3.forward(l2.forward(l1.forward(input))); // forward propogation

// FUNCTIONS
// lol, i watched 1 cs50 course video, and now i abstract EVERYTHING lmao