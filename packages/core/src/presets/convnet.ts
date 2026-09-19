/**
 * Convolutional classifiers.
 *
 * The other shape of network this tool can draw, and the one it was not built
 * for. A convnet has no sequence: its tensors are `B C H W`, its spatial extent
 * shrinks layer by layer, and its "token" is one image, which is what `T = 1`
 * means in these designs. Everything the analysis reports still holds with that
 * reading — FLOPs per image rather than per token, activations per image,
 * and a KV cache of zero because there is nothing to cache.
 *
 * This is written out block by block rather than generated from a spec. AlexNet
 * is one specific network with five convolutions of five different shapes, and
 * a builder that took eleven arguments to describe one model would be a worse
 * description than the model itself.
 */

import type { Doc, Graph, NodeDef, SymbolDef } from "../ir/types.js";
import { DOC_VERSION } from "../ir/types.js";

/**
 * AlexNet, as torchvision builds it.
 *
 * The 2012 paper split the network across two GPUs with half the channels on
 * each and cross-connections only at certain layers, because a GTX 580 had
 * 3 GB. Everyone since has used the single-stream form from Krizhevsky's 2014
 * follow-up, which is what `torchvision.models.alexnet` is and what the
 * 61,100,840-parameter figure refers to. That is the one drawn here.
 *
 * The number worth looking at once it is on screen: the five convolutions come
 * to 2.5M parameters and the three fully-connected layers to 58.6M. Ninety-six
 * per cent of the weights are in the classifier, and almost all of that is the
 * first one, flattening 9,216 activations into 4,096. The convolutions do most
 * of the arithmetic and hold almost none of the memory, which is the tension
 * every architecture since has been arguing with.
 */
export function alexnet(): Doc {
  const symbols: Record<string, SymbolDef> = {
    B: { kind: "runtime", default: 1, doc: "Images per batch" },
    // A convnet's token is an image. Keeping T at one lets every per-token
    // number the analysis reports be read as per-image without changing it.
    T: { kind: "runtime", default: 1, doc: "One image per sample; a convnet has no sequence" },
    Ccls: { kind: "design", value: 1000, doc: "ImageNet-1k classes" },
    Res: { kind: "design", value: 224, doc: "Input resolution" },
    Fc: { kind: "design", value: 4096, doc: "Width of the fully-connected layers" },
  };

  const nodes: NodeDef[] = [
    { id: "image", type: "input", label: "Image", params: { shape: "B 3 224 224", dtype: "fp32" } },

    // --- the convolutional process -----------------------------------------
    {
      id: "conv1",
      type: "conv2d",
      label: "Conv 1",
      // 11x11 at stride 4: the one enormous first kernel that later networks
      // replaced with a stack of 3x3s.
      params: { in_channels: 3, out_channels: 64, kernel: 11, stride: 4, padding: 2, in_h: 224, in_w: 224, act: "relu" },
    },
    { id: "pool1", type: "maxpool2d", label: "Pool 1", params: { channels: 64, kernel: 3, stride: 2, in_h: 55, in_w: 55 } },
    {
      id: "conv2",
      type: "conv2d",
      label: "Conv 2",
      params: { in_channels: 64, out_channels: 192, kernel: 5, stride: 1, padding: 2, in_h: 27, in_w: 27, act: "relu" },
    },
    { id: "pool2", type: "maxpool2d", label: "Pool 2", params: { channels: 192, kernel: 3, stride: 2, in_h: 27, in_w: 27 } },
    {
      id: "conv3",
      type: "conv2d",
      label: "Conv 3",
      params: { in_channels: 192, out_channels: 384, kernel: 3, stride: 1, padding: 1, in_h: 13, in_w: 13, act: "relu" },
    },
    {
      id: "conv4",
      type: "conv2d",
      label: "Conv 4",
      params: { in_channels: 384, out_channels: 256, kernel: 3, stride: 1, padding: 1, in_h: 13, in_w: 13, act: "relu" },
    },
    {
      id: "conv5",
      type: "conv2d",
      label: "Conv 5",
      params: { in_channels: 256, out_channels: 256, kernel: 3, stride: 1, padding: 1, in_h: 13, in_w: 13, act: "relu" },
    },
    { id: "pool3", type: "maxpool2d", label: "Pool 3", params: { channels: 256, kernel: 3, stride: 2, in_h: 13, in_w: 13 } },

    // --- the classification process ----------------------------------------
    { id: "flatten", type: "flatten2d", label: "Flatten", params: { channels: 256, in_h: 6, in_w: 6 } },
    { id: "fc1", type: "linear", label: "FC 1", params: { in_features: 9216, out_features: "Fc", bias: true } },
    { id: "act1", type: "activation", params: { kind: "relu", dim: "Fc" } },
    { id: "fc2", type: "linear", label: "FC 2", params: { in_features: "Fc", out_features: "Fc", bias: true } },
    { id: "act2", type: "activation", params: { kind: "relu", dim: "Fc" } },
    { id: "fc3", type: "linear", label: "Classifier", params: { in_features: "Fc", out_features: "Ccls", bias: true } },
    { id: "scores", type: "output", label: "Class scores" },
  ];

  const chain = [
    "image:x",
    "conv1:x",
    "conv1:y",
    "pool1:x",
    "pool1:y",
    "conv2:x",
    "conv2:y",
    "pool2:x",
    "pool2:y",
    "conv3:x",
    "conv3:y",
    "conv4:x",
    "conv4:y",
    "conv5:x",
    "conv5:y",
    "pool3:x",
    "pool3:y",
    "flatten:x",
    "flatten:y",
    "fc1:x",
    "fc1:y",
    "act1:x",
    "act1:y",
    "fc2:x",
    "fc2:y",
    "act2:x",
    "act2:y",
    "fc3:x",
    "fc3:y",
    "scores:x",
  ];
  const edges: Graph["edges"] = [];
  for (let i = 0; i < chain.length; i += 2) edges.push([chain[i], chain[i + 1]]);

  return {
    version: DOC_VERSION,
    meta: {
      name: "alexnet",
      family: "convnet",
      notes:
        "AlexNet as torchvision builds it: the single-stream form from Krizhevsky's 2014 follow-up, " +
        "not the two-GPU split of the 2012 paper. Five convolutions come to 2,469,696 parameters and " +
        "the three fully-connected layers to 58,631,144 — ninety-six per cent of the weights are in " +
        "the classifier, and three quarters of those are in the single layer that flattens 9,216 " +
        "activations into 4,096. Dropout and the adaptive average pool are omitted: neither has " +
        "parameters, and at 224px the pool is the identity. This is also the only design here whose " +
        "tensors are not a sequence, so its token is one image and T is 1.",
      published: {
        params: 61_100_840,
        source: "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html",
      },
    },
    symbols,
    graph: { nodes, edges },
    ui: {},
  };
}
