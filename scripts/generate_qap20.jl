#!/usr/bin/env julia
# Stream the QAP20 LP representation used by the paper, preserving row and
# column order, explicit bound rows, and empty self-symmetry equalities.

function read_dat(path)
    tokens = split(read(path, String))
    n = parse(Int, tokens[1])
    n >= 2 || error("QAP dimension must be at least two")
    length(tokens) == 1 + 2n^2 || error("Expected N and two N-by-N matrices")
    values = parse.(Int, tokens[2:end])
    flow = Matrix(reshape(values[1:n^2], n, n)')
    distance = Matrix(reshape(values[n^2+1:end], n, n)')
    return n, flow, distance
end

# JuMP arrays in the original exports use column-major variable order.
idx2(n, i, j) = i + n*(j-1)
idx3(n, i, j, k) = i + n*(j-1) + n^2*(k-1)
idx4(n, i, j, k, l) = i + n*(j-1) + n^2*(k-1) + n^3*(l-1)
card(io, column, row, value) = println(io, "    ", column, " ", row, " ", value)

function write_mps(destination, n, flow, distance)
    # Batch writes to the gzip pipe; printing each field directly to a pipe
    # would issue millions of small system calls for a benchmark instance.
    io = IOBuffer(sizehint=1 << 20)
    function flush_buffer()
        if position(io) >= 1 << 20
            write(destination, take!(io))
        end
    end
    n2, n3, n4 = n^2, n^3, n^4
    neq = 2n + 2n3 + n4
    println(io, "NAME QAP20\nROWS\n N OBJ")
    for q in 1:n4
        println(io, " L c", q, "_1")  # -y <= 0
        flush_buffer()
    end
    for r in 1:neq
        println(io, " E c", r)
        flush_buffer()
    end
    for q in 1:n2
        println(io, " L c", q, "_2")  # 0 <= x <= 1, via RANGES
        flush_buffer()
    end
    println(io, "COLUMNS")
    for j in 1:n, i in 1:n
        column = "x[$i,$j]"
        card(io, column, "c$i", 1)
        card(io, column, "c$(n+j)", 1)
        for t in 1:n
            card(io, column, "c$(2n + idx3(n,t,i,j))", -1)
        end
        for t in 1:n
            card(io, column, "c$(2n + n3 + idx3(n,t,i,j))", -1)
        end
        card(io, column, "c$(idx2(n,i,j))_2", 1)
        flush_buffer()
    end
    for l in 1:n, k in 1:n, j in 1:n, i in 1:n
        column = "y[$i,$j,$k,$l]"
        q = idx4(n,i,j,k,l)
        reverse_q = idx4(n,k,l,i,j)
        card(io, column, "c$(q)_1", -1)
        card(io, column, "c$(2n + idx3(n,j,k,l))", 1)
        card(io, column, "c$(2n + n3 + idx3(n,i,k,l))", 1)
        if q != reverse_q
            first, second = minmax(q, reverse_q)
            sign = q < reverse_q ? 1 : -1
            card(io, column, "c$(2n+2n3+first)", sign)
            card(io, column, "c$(2n+2n3+second)", -sign)
        end
        cost = flow[i,k] * distance[j,l]
        cost != 0 && card(io, column, "OBJ", cost)
        flush_buffer()
    end
    println(io, "RHS")
    for q in 1:n4
        card(io, "rhs", "c$(q)_1", 0)
        flush_buffer()
    end
    for r in 1:neq
        card(io, "rhs", "c$r", r <= 2n ? 1 : 0)
        flush_buffer()
    end
    for q in 1:n2
        card(io, "rhs", "c$(q)_2", 1)
        flush_buffer()
    end
    println(io, "RANGES")
    for q in 1:n2
        card(io, "rhs", "c$(q)_2", 1)
        flush_buffer()
    end
    println(io, "BOUNDS")
    for j in 1:n, i in 1:n
        println(io, " FR bounds x[$i,$j]")
        flush_buffer()
    end
    for l in 1:n, k in 1:n, j in 1:n, i in 1:n
        println(io, " FR bounds y[$i,$j,$k,$l]")
        flush_buffer()
    end
    println(io, "ENDATA")
    write(destination, take!(io))
end

function main()
    length(ARGS) == 2 || error("Usage: julia scripts/generate_qap20.jl INPUT.dat OUTPUT.mps.gz")
    input, output = ARGS
    ispath(output) && error("Refusing to overwrite $output")
    n, flow, distance = read_dat(input)
    println("Generating $(basename(output)): N=$n, rows=$(2n^4 + 2n^3 + n^2 + 2n), columns=$(n^4 + n^2)")
    flush(stdout)
    mkpath(dirname(abspath(output)))
    temporary = output * ".part-$(getpid())"
    try
        if endswith(output, ".gz")
            open(pipeline(`gzip -n -c`, stdout=temporary), "w") do io
                write_mps(io, n, flow, distance)
            end
        else
            open(temporary, "w") do io
                write_mps(io, n, flow, distance)
            end
        end
        mv(temporary, output)
    finally
        isfile(temporary) && rm(temporary)
    end
    println("Created $output (N=$n)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
