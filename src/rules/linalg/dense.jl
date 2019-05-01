#####
##### `sum`
#####

frule(::typeof(sum), x) = (sum(x), Rule(sum))

rrule(::typeof(sum), x) = (sum(x), Rule(cast))

#####
##### `dot`
#####

function frule(::typeof(dot), x, y)
    return dot(x, y), Rule((Δx, Δy) -> sum(Δx * cast(y)) + sum(cast(x) * Δy))
end

function rrule(::typeof(dot), x, y)
    return dot(x, y), (Rule(ΔΩ -> ΔΩ * cast(y)), Rule(ΔΩ -> cast(x) * ΔΩ))
end

#####
##### `inv`
#####

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    return Ω, Rule(Δx -> m * Δx * Ω)
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω')
    return Ω, Rule(ΔΩ -> m * ΔΩ * Ω')
end

#####
##### `det`
#####

function frule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x))
    return Ω, Rule(Δx -> Ω * tr(extern(m * Δx)))
end

function rrule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x)')
    return Ω, Rule(ΔΩ -> Ω * ΔΩ * m)
end

#####
##### `logdet`
#####

function frule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x))
    return Ω, Rule(Δx -> tr(extern(m * Δx)))
end

function rrule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x)')
    return Ω, Rule(ΔΩ -> ΔΩ * m)
end

#####
##### `trace`
#####

frule(::typeof(tr), x) = (tr(x), Rule(Δx -> tr(extern(Δx))))

rrule(::typeof(tr), x) = (tr(x), Rule(ΔΩ -> Diagonal(fill(ΔΩ, size(x, 1)))))

#####
##### Binary operations
#####

const BINARY_LINALG_OPS = [
    (*, AbstractArray, AbstractArray,
        :(Ȳ * B'),
        :(A' * Ȳ)),
    (*, Transpose, AbstractArray,
        :(B * transpose(Ȳ)),
        :(A * Ȳ)),
    (*, AbstractArray, Transpose,
        :(Ȳ * B),
        :(transpose(Ȳ) * A)),
    (*, Transpose, Transpose,
        :(transpose(B) * transpose(Ȳ)),
        :(transpose(Ȳ) * transpose(A))),
    (*, Adjoint, AbstractArray,
        :(B * transpose(Ȳ)),
        :(A * Ȳ)),
    (*, AbstractArray, Adjoint,
        :(Ȳ * B),
        :(Ȳ' * A)),
    (*, Adjoint, Adjoint,
        :(B' * Ȳ'),
        :(Ȳ' * A')),
    (/, AbstractArray, AbstractArray,
        :(Ȳ / transpose(B)),
        :(-transpose(Y) * (Ȳ / transpose(B)))),
    (/, Transpose, AbstractArray,
        :(B \ transpose(Ȳ)),
        :(-transpose(Y) * (Ȳ / transpose(B)))),
    (/, AbstractArray, Transpose,
        :(Ȳ / B),
        :(-(transpose(B) \ transpose(Ȳ)) * Y)),
    (/, Transpose, Transpose,
        :(transpose(B) \ transpose(Ȳ)),
        :(-(transpose(B) \ transpose(Ȳ)) * Y)),
    (/, Adjoint, AbstractArray,
        :(B \ Ȳ'),
        :(-Y' * (Ȳ / B'))),
    (/, AbstractArray, Adjoint,
        :(Ȳ / B),
        :(-(transpose(B) \ transpose(Ȳ)) * Y)),
    (/, Adjoint, Adjoint,
        :(B' \ Ȳ'),
        :(-(B' \ Ȳ') * Y)),
    (\, AbstractArray, AbstractArray,
        :(-(transpose(A) \ Ȳ) * transpose(Y)),
        :(transpose(A) \ Ȳ)),
    (\, Transpose, AbstractArray,
        :(-Y * transpose(A \ Ȳ)),
        :(A \ Ȳ)),
    (\, AbstractArray, Transpose,
        :(-transpose(transpose(Ȳ) / A) * transpose(Y)),
        :(transpose(Ȳ) / A)),
    (\, Transpose, Transpose,
       :(-Y * (transpose(Ȳ) / transpose(A))),
       :(transpose(Ȳ) / transpose(A))),
    (\, Adjoint, AbstractArray,
        :(-Y * (A \ Ȳ)'),
        :(A \ Ȳ)),
    (\, AbstractArray, Adjoint,
        :(-(Ȳ' / A)' * Y),
        :(Ȳ' / A)),
    (\, Adjoint, Adjoint,
        :(-Y * (Ȳ' / A')),
        :(Ȳ' / A')),
    #(norm, AbstractArray, Number,
    #    :(Ȳ .* Y^(1 - B) .* abs.(A).^B ./ A),
    #    :(Ȳ * (Y^(1 - B) * sum(abs.(A).^B .* log.(abs.(A))) - Y * log(Y)) / B)),
    #(norm, Number, Number,
    #    :(Ȳ * sign(A)),
    #    :(zero(A) + zero(B))),
]

for (f, TA, TB, Ā, B̄) in BINARY_LINALG_OPS
    @eval function rrule(::typeof($f), A::$TA, B::$TB)
        Y = $f(A, B)
        ∂A = Rule(Ȳ -> $Ā)
        ∂B = Rule(Ȳ -> $B̄)
        return Y, (∂A, ∂B)
    end
end
