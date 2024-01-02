using Gen
using Plots

@gen function polyas_urn_model(observations::Vector{Symbol})
    # Assume initial counts of red and blue balls are random variables to be inferred
    red_count = @trace(uniform_discrete(1, 20), :red_count)  # Example prior: Uniform between 1 and 20
    blue_count = @trace(uniform_discrete(1, 20), :blue_count)  # Example prior: Uniform between 1 and 20

    reinforcement = @trace(uniform_discrete(0, 5), :reinforcement)  # Reinforcement value

    for i in 1:length(observations)
        total_balls = red_count + blue_count
        p_red = red_count / total_balls
        p_blue = blue_count / total_balls

        probs = [p_red, p_blue]
        observed_color = @trace(categorical(probs), (:ball, i))

        # Update counts based on observed color
        if observed_color == 1  # Red is the first category
            red_count += reinforcement
        else  # Blue is the second category
            blue_count += reinforcement
        end

        # Check if the observed color matches the actual drawn color
        if observations[i] == :red
            @trace(bernoulli(p_red), (:observation, i))
        else
            @trace(bernoulli(p_blue), (:observation, i))
        end
    end

    return (red_count, blue_count, reinforcement)
end

observations = [:red, :red, :red, :red, :red]  # Example observations
inferred_ratios = []

# Run the model and inference
for _ in 1:1000
    trace = Gen.simulate(polyas_urn_model, (observations,))
    red_count = trace[:red_count]
    blue_count = trace[:blue_count]
    # Calculate and store the ratio of red to blue counts
    push!(inferred_ratios, red_count / blue_count)
end

# Plotting the histogram of inferred red to blue ratios
histogram(inferred_ratios, bins=30, xlabel="Red/Blue Ratio", ylabel="Frequency", label="Inferred Ratios")
savefig("inferred_ratios.png")
