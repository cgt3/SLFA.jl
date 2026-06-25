using SLFA
using Documenter

DocMeta.setdocmeta!(SLFA, :DocTestSetup, :(using SLFA); recursive=true)

makedocs(;
    modules=[SLFA],
    authors="Christina Taylor, Braden King",
    sitename="SLFA.jl",
    format=Documenter.HTML(;
        canonical="https://cgt3.github.io/SLFA.jl",
        edit_link=get(ENV, "GITHUB_REF_TYPE", "") == "tag" ? :commit : "dev",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cgt3/SLFA.jl",
    devbranch="dev",
    devurl="dev"
)
