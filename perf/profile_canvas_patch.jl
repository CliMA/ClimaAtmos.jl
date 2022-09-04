import JSON, ProfileCanvas, Profile
# Temporary until this merges:
#    https://github.com/pfitzseb/ProfileCanvas.jl/pull/18
html_file(filename, data = Profile.fetch(); kwargs...) =
    html_file(filename, ProfileCanvas.view(data; kwargs...))

function html_file(file::AbstractString, canvas::ProfileCanvas.ProfileData)
    @assert endswith(file, ".html")
    open(file, "w") do io
        id = "profiler-container-$(round(Int, rand()*100000))"

        println(
            io,
            """
            <html>
            <head>
            <style>
                #$(id) {
                    margin: 0;
                    padding: 0;
                    width: 100vw;
                    height: 100vh;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
                    overflow: hidden;
                }
                body {
                    margin: 0;
                    padding: 0;
                }
            </style>
            </head>
            <body>
                <div id="$(id)"></div>
                <script type="module">
                    const ProfileCanvas = await import('$(ProfileCanvas.jlprofile_data_uri())')
                    const viewer = new ProfileCanvas.ProfileViewer("#$(id)", $(JSON.json(canvas.data)), "$(canvas.typ)")
                </script>
            </body>
            </html>
            """,
        )
    end
    return file
end

# This seems like it should work in theory,
# but the window/view is somehow very truncated.
function html_file_sprint(
    file::AbstractString,
    data = Profile.fetch();
    kwargs...,
)
    trace = ProfileCanvas.view(data; kwargs...)
    open(file, "w") do io
        print(io, sprint(show, "text/html", trace))
    end
end
