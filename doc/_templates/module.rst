{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}

   Functions
   ---------

   {% for item in functions %}
   .. autofunction:: {{ item }}

   .. raw:: html

	       <div style='clear:both'></div>
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block exceptions %}
   {% if exceptions %}


   Exceptions
   ----------

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
