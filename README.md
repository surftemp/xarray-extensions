# xarray-extensions

library to add useful functions to xarray

## Tools - Analysis

To use these, clone this repo and place the root directory on your python path.  

Then import the module into your code using `import xarray_extensions`.  Doing so will attach the following extra methods to xarray DataArray and Dataset objects:

         
  <section id="dataarray-methods-timeseries">
<h1>DataArray methods - timeseries<a class="headerlink" href="#dataarray-methods-timeseries" title="Permalink to this headline">¶</a></h1>
<p>xarray 0.19.0 or later is recommended.  These methods extend xarray with timeseries functionality.</p>
<p>include <cite>import xarray_extensions.timeseries</cite> in your code to add these methods to xarray.DataArray</p>
<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.timeseries.deseasonalised">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">deseasonalised</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">clim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">abs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.deseasonalised" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain a deseasonalised DataArray based on a monthly climatology</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound</p></li>
<li><p><strong>clim</strong> (<em>boolean</em>) – if True, return the climatology</p></li>
<li><p><strong>abs</strong> (<em>boolean</em>) – if False, return monthly anomalies, otherwise return the anomalies plus the monthly means from the climatology</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module is imported</p>
</dd></dl>

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.timeseries.detrended">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">detrended</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">quadratic</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">coeff</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">coeff_scale</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">'year'</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">abs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.detrended" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain a detrended DataArray using a linear or quadratic function to fit a trend</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound</p></li>
<li><p><strong>quadratic</strong> (<em>boolean</em>) – if True, fit a quadratic (degree=2) model, otherwise fit a linear (degree=1) model, along the time axis</p></li>
<li><p><strong>coeff</strong> (<em>boolean</em>) – if True, return the model coefficients</p></li>
<li><p><strong>coeff_scale</strong> (<em>str</em>) – set coefficients to work on a particular timescale, should be one of “year”, “day”, “second”.
set to None to scale using nanoseconds (xarray’s default representation)</p></li>
<li><p><strong>abs</strong> (<em>boolean</em>) – if False, return differences between the original values and the model values, otherwise return the differences
plus the model means</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module is imported</p>
</dd></dl>

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.timeseries.lagged_correlation">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">lagged_correlation</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">otherda</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">lags</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.lagged_correlation" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain pearson correlation coefficients between this DataArray and another DataArray</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound, assumed to include a “time” dimension</p></li>
<li><p><strong>otherda</strong> (<em>xarray.DataArray</em>) – the other DataArray against which the correlation is to be performed, assumed to have dimensions (time)</p></li>
<li><p><strong>lags</strong> (<em>list</em><em>[</em><em>int</em><em>]</em>) – a list of lags to apply to the other dataset before calculating the correlation coefficient</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance having dimensions (lags,lat,lon)</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module is imported</p>
</dd></dl>

</section>
<section id="dataset-methods-general">
<h1>Dataset Methods - general<a class="headerlink" href="#dataset-methods-general" title="Permalink to this headline">¶</a></h1>
<p>include <cite>import xarray_extensions.general</cite> in your code to add these methods to xarray.Dataset</p>
<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.general.safe_assign">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.general.</span></span><span class="sig-name descname"><span class="pre">safe_assign</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">da</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">name</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">''</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.general.safe_assign" title="Permalink to this definition">¶</a></dt>
<dd><p>Attach a DataArray as a variable to a Dataset, working around a bug in xarray when using simple assignment
See <a class="reference external" href="https://github.com/pydata/xarray/issues/2245">https://github.com/pydata/xarray/issues/2245</a> (dimension attributes are sometimes lost)</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.Dataset</em>) – the DataSet instance to which the DataArray is to be assigned</p></li>
<li><p><strong>da</strong> (<em>xarray.DataArray</em>) – the DataArray to assign</p></li>
<li><p><strong>name</strong> (<em>str</em>) – the name of the variable to assign.  If empty, uses the name of the DataArray.</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataSet class as a method when this module is imported</p>
</dd></dl>

</section>


