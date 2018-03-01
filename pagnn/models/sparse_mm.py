import torch
from torch.autograd import Function


class SparseMM(Function):

    @staticmethod
    def forward(ctx, matrix1, matrix2):
        ctx.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    @staticmethod
    def backward(ctx, grad_output):
        matrix1, matrix2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if ctx.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class LeftMatMulSparseFixedWeights(Function):
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable,
    returning a Dense one. This is added because there's no autograd for sparse yet.

    No gradient computed on the sparse weights.

    References:
        * https://discuss.pytorch.org/t/autograd-for-sparse-matmul-getting-either-cuda-memory-\
leak-or-buffers-have-already-been-freed-error/3806
    """

    @staticmethod
    def forward(ctx, matrix1, matrix2):
        ctx.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    @staticmethod
    def backward(ctx, grad_output):
        matrix1, matrix2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if ctx.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2
