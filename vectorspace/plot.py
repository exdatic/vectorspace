__all__ = ['plot_scatter']


def plot_vega(spec, renderer='canvas'):
    import json
    from string import Template
    from IPython.display import display, Javascript

    js = Template("""
    (function(element) {
        requirejs.config({
          baseUrl: "https://cdn.jsdelivr.net/npm/",
          paths: {
            "vega-embed": "vega-embed@6?noext",
            "vega-lite": "vega-lite@5?noext",
            "vega": "vega@5?noext"
          }
        });
        require(['vega-embed'], function(vegaEmbed) {
          vegaEmbed(element.get(0), $spec, {
              renderer: "$renderer",
              export: true,
              source: true
          }).catch(console.warn);
        });
    })(element);
    """).substitute(spec=json.dumps(spec), renderer=renderer)

    display(Javascript(js))


def plot_scatter(x, y, names=None, colors=None, width=500, height=300, spec_file='zoomable_scatter.json'):
    import pkg_resources
    import json

    spec = json.loads(pkg_resources.resource_string('vectorspace', spec_file))
    spec['width'] = width
    spec['height'] = height
    if names is not None:
        data = [dict(u=float(u), v=float(v), name=name) for u, v, name in zip(x, y, names)]
    else:
        for i, mark in enumerate(list(spec['marks'])):
            if 'type' in mark and mark['type'] == 'label':
                del spec['marks'][i]
        data = [dict(u=float(u), v=float(v)) for u, v in zip(x, y)]
    if colors is not None:
        data = [dict(**item, color=color) for item, color in zip(data, colors)]
    else:
        data = [dict(**item, color=color) for item, color in zip(data, ['steelblue'] * len(data))]
    spec['data'][0]['values'] = data
    plot_vega(spec)
