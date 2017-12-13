{%- extends 'full.tpl' -%}

<!-- Begin inliner.tpl -->
{%- block html_head -%}

{{ super() }}

{% for css in resources.inliner.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

{% for js in resources.inliner.js -%}
    <script type="text/javascript">
    {{ js }}
    </script>
{% endfor %}

{%- endblock html_head -%}
<!-- End inliner.tpl -->


<!-- Begin collapsible_headings.tpl -->
{% block markdowncell scoped %}
{%- if cell.metadata.heading_collapsed -%}
<div class="cell border-box-sizing text_cell rendered collapsible_headings_collapsed">
{%- else -%}
<div class="cell border-box-sizing text_cell rendered">
{%- endif -%}
{{ self.empty_in_prompt() }}
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
{{ cell.source  | markdown2html | strip_files_prefix }}
</div>
</div>
</div>
{%- endblock markdowncell %}
<!-- End collapsible_headings.tpl -->


<!-- Begin toggle.tpl -->
{% block input_group %}
{{ super() }}
{% endblock input_group %}

{% block output_group %}
<div class="output_hidden">
{{ super() }}
</div>
{% endblock output_group %}
<!-- End toggle.tpl -->


{%- block header -%}
{{ super() }}

<script src="//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>

<style type="text/css">
div.text_cell_render {
  margin-top: 0px;
}
.output_hidden {
  display: none;
  margin-top: 5px;
}
</style>

<script>
$(document).ready(function(){
  $(".input").click(function(){
      $(this).next('.output_hidden').slideToggle();
  });
})
</script>
{%- endblock header -%}
