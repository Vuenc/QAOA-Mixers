using Test
using TestSetExtensions


@testset "All the tests" begin
    @includetests ARGS
    # @includetests ["test_register"]
end
