// the plan is to create a basic, regressional mlp which approximates a function.
// the constraint is that i'm not allowed to use ANY external libraries
// hi github
// i wrote comments just for you

const nonLinearities = {
    ReLU: {
        fw: (x) => Math.max(0, x),
        bw: (x) => x > 0 ? 1 : 0
    },
}

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
                let dotProduct = parameters[parameters.length - 1]; // set it to bias first so that i dont have to add it later
                for (let i = 0; i < parameters.length - 1; i++) {
                    dotProduct = dotProduct + (parameters[i] * input[i]);
                }

                dotProduct = this.nonLinearity ? this.nonLinearity.fw(dotProduct) : dotProduct;

                output.push(dotProduct);
            }

            this.input = input;
            this.output = output;
            this.fanIn = inputDimension;
            this.fanOut = neuronParameters.length;

            return output;
        } else {
            console.error('chud alert: quotient didnt match input dimension. quotient: ', quotient, 'input dim: ', inputDimension);
        }

    }

    backward(nextGradient) { //remember this would be for an entire layer
        if (this.input) {

            // backprop thru a layer is a matrix multiplication of the next gradient and the jacobian of the layer, but we just dont do it formally
            // remember than fan in of a layer = fan out of the layer before it. 
            // a gradient has the same shape as what its differentiating (shape of nabla theta = shape of theta)
            // relearn the math later

            // make a gradient for parameters and for input;
            let gradient = {
                wrtTheta: null,
                wrtInput: null,
            }

            const layerInput = [...this.input, 1]; // make space for bias via pushing 1

            gradient.wrtTheta = Array.from(this.parameters).map((_, i) => {
                const aFGrad = this.nonLinearity ? this.nonLinearity.bw(this.output[i / layerInput.length]) : 1; // get the bw nonlinearity of the current neuron
                return layerInput[i % layerInput.length] * nextGradient[Math.floor(i / layerInput.length)] * aFGrad;
            });

            gradient.wrtInput = Array.from(this.input).map((_, i) => {
                let v = 0;
                for (let j = 0; j < this.fanOut; j++) { // for each neuron
                    const aFGrad = this.nonLinearity ? this.nonLinearity.bw(this.output[j]) : 1; // get the bw nonlinearity of the current neuron
                    const weightIdx = j * layerInput.length + i; // weight from input i to neuron j
                    v += this.parameters[weightIdx] * nextGradient[j] * aFGrad;
                }
                return v;
            });

            return gradient;


        } else {
            console.warn('hey, run the network first you chud');
        }
    }

    destroy() {
        
    }
}

class NeuralNetwork {
    constructor(layers) { // ouu shii i forgot that getting device n adapter is async so i move to init method
        this.device = null;
        this.layers = layers;

        // layers and stuff go here
    }

    // BUT, i wont touch GPU right now, too little time before exams to introduce a new system that i dont understand as much

    // async init() { // initialise webgpu stff
    //     const gpu = navigator.gpu;
    //     if (gpu) { //icl im just using an if statement and nesting everything in there bc it looks cooler

    //         const adapter = await gpu.requestAdapter();
    //         this.device = await adapter.requestDevice();

    //         // then i set the batches and stuff and the pipeline.... later. not here

    //     } else {
    //         console.error("no gpu found")
    //     }
    // }

    run(x) {
        let workspace = [x];
        for (let i = 0; i < this.layers.length; i++) {
            workspace.push(this.layers[i].forward(workspace[i]));
        }
        console.log(workspace);
    }

    backOnce(lossGradient) {
        let currentGradient = lossGradient; // start with loss gradient array
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const grad = this.layers[i].backward(currentGradient);
            currentGradient = grad.wrtInput; // pass output gradient to previous layer
        }
    }

    destroy() {

    }
}

// and so running should look something like

// const layer = new Layer([1, 2, 1, 1, 5, 2]);
// layer.forward([1, 3]);
// layer.backward();

const nn = new NeuralNetwork([
    new Layer([-3, -2, 0, -5, -1, -5, -3, -6], nonLinearities.ReLU), // 2 neurons, 3 weights e
    new Layer([-1, -3, -5, -2, 6, 3, -2, 0, 1], nonLinearities.ReLU), // 3 neurons, 2 weights e
    new Layer([3, -1, 2, 1]) // 2 neurons, 1 weight
])

nn.run([1, 5, 3]);
nn.backOnce([2]);


// const input; //whatever input would be (user selected)
// const [l1, l2, l3] = [new Layer(), new Layer(), new Layer()];

// // compositional function form
// l3.forward(l2.forward(l1.forward(input))); // forward propogation