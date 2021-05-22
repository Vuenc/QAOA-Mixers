using Test
using TestSetExtensions


@testset "All the tests" begin
    @includetests ["test_adjoints", "test_gradients", "test_hamiltonians", "test_hamming_weight", "test_parity_partition_mixers"]
end
