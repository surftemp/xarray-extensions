# xarray-extensions

library to add useful functions to xarray

## Tools - Analysis

To use these, clone this repo and place the root directory on your python path.  

Then import the module into your code using 

```
import xarray_extensions.timeseries  # for timeseries extensions
import xarray_extensions.general     # for general extensions
```

Doing so will attach the following extra methods to xarray DataArray and Dataset objects:

 <p>xarray 0.19.0 or later is required, attempting to import these extensions with an earlier version of xarray will cause
an exception to the thrown.  These methods extend xarray with timeseries and various other functionality.</p>
<section id="dataarray-methods-timeseries">
<h1>DataArray methods - timeseries<a class="headerlink" href="#dataarray-methods-timeseries" title="Permalink to this headline">¶</a></h1>
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
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">lagged_correlation</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">otherda</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">lags</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">ci</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">dof</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.lagged_correlation" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain pearson correlation coefficients between this DataArray and another DataArray, with a series of lags applied
return the correlation coefficients</p>
<p>if significance threshold is to be calculated, return an extra parameter dimension with the correlation at index 0 and
the two-tailed significance threshold at index 1</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound, assumed to include a “time” dimension</p></li>
<li><p><strong>otherda</strong> (<em>xarray.DataArray</em>) – the other DataArray against which the correlation is to be performed, assumed to have a “time” dimensions</p></li>
<li><p><strong>lags</strong> (<em>list</em><em>[</em><em>int</em><em>]</em>) – a list of lags to apply to the time dimension before calculating the correlation coefficient
where lag=L means that we correlate self[L:,…] with otherda[:-L,…]</p></li>
<li><p><strong>ci</strong> (<em>float</em>) – specify the confidence interval if significance is to be calculated (for example, specify 0.05 for 95% threshold)</p></li>
<li><p><strong>dof</strong> (<em>int</em>) – set the degrees of freedom manually
(TODO, if not specified this should be computed from the data, currently return NaN)</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
dimension</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
<dt class="field-even">Raises</dt>
<dd class="field-even"><p><strong>MisalignedTimeAxisException</strong> – if this array and the other array do not have identical time coordinates</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
is imported.</p>
<p class="rubric">Example</p>
<p>A way you might use me is:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">sla</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SLA</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>
<span class="n">sst</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SST</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>

<span class="c1"># get the correlations between SLAs and SSTs that occur one month earlier</span>
<span class="n">correlation</span> <span class="o">=</span> <span class="n">sla</span><span class="o">.</span><span class="n">lagged_correlation</span><span class="p">(</span><span class="n">sst</span><span class="p">,</span> <span class="n">lags</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
</pre></div>
</div>
</dd></dl>

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.timeseries.lagged_correlation_month_of_year">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">lagged_correlation_month_of_year</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">otherda</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">lags</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">month_of_year</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">ci</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">dof</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.lagged_correlation_month_of_year" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain pearson correlation coefficients between a yearly timeseries extracted from this DataArray at a particular month
and those from another DataArray, with a series of lags applied return the correlation coefficients</p>
<p>if significance threshold is to be calculated, return an extra parameter dimension with the correlation at index 0 and
the two-tailed significance threshold at index 1</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the main DataArray instance (to which this method is bound), assumed to include a “time” dimension</p></li>
<li><p><strong>otherda</strong> (<em>xarray.DataArray</em>) – the other DataArray against which the correlation is to be performed, assumed to have a “time” dimensions</p></li>
<li><p><strong>lags</strong> (<em>list</em><em>[</em><em>int</em><em>]</em>) – a list of lags to apply to the time dimension before calculating the correlation coefficient
where lag=L means that we correlate self[L:,…] with otherda[:-L,…]</p></li>
<li><p><strong>month_of_year</strong> (<em>int</em>) – indicate which month (1=jan, 2=feb, etc) to analyse in the main DataArray</p></li>
<li><p><strong>ci</strong> (<em>float</em>) – specify the confidence interval if significance is to be calculated (for example, specify 0.05 for 95% threshold)</p></li>
<li><p><strong>dof</strong> (<em>int</em>) – set the degrees of freedom manually
(TODO, if not specified this should be computed from the data, currently return NaN)</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
dimension</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
<dt class="field-even">Raises</dt>
<dd class="field-even"><ul class="simple">
<li><p><strong>MisalignedTimeAxisException</strong> – if this array and the other array do not have identical time coordinates</p></li>
<li><p><strong>InvalidMonthOfYearException</strong> – if the month_of_year parameter is not an integer in the range 1-12</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
is imported.</p>
<p class="rubric">Example</p>
<p>A way you might use me is:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">sla</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SLA</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>
<span class="n">sst</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SST</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>

<span class="c1"># get the correlations between april SLAs and march SSTs over successive years</span>
<span class="n">correlation</span> <span class="o">=</span> <span class="n">sla</span><span class="o">.</span><span class="n">lagged_correlation_month_of_year</span><span class="p">(</span><span class="n">sst</span><span class="p">,</span> <span class="n">lags</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">month_of_year</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span>
</pre></div>
</div>
</dd></dl>

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.timeseries.lagged_regression">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">lagged_regression</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">otherda</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">lags</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.lagged_regression" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain linear regression coefficients between this DataArray and another DataArray, with a series of lags applied.
The other DataArray is treated as the y variable, this DataArray is treated as the x variable and the coefficients
returned are the values [m,c] from y = mx+c</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound, assumed to include a “time” dimension</p></li>
<li><p><strong>otherda</strong> (<em>xarray.DataArray</em>) – the other DataArray against which the correlation is to be performed, assumed to have dimensions (time)</p></li>
<li><p><strong>lags</strong> (<em>list</em><em>[</em><em>int</em><em>]</em>) – a list of lags to apply to the time dimension before calculating the regression coefficients
where lag=L means that we compute the regression coefficients for x=self[L:,…] and y=otherda[:-L,…]</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
dimension and an extra parameter dimension added (with size 2, where index 0 holds the slope value m and index 1
holds the intercept value c)</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
<dt class="field-even">Raises</dt>
<dd class="field-even"><p><strong>MisalignedTimeAxisException</strong> – if this array and the other array do not have identical time coordinates</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
is imported.</p>
<p class="rubric">Example</p>
<p>A way you might use me is:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">sla</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SLA</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>
<span class="n">sst</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SST</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>

<span class="c1"># get the regression coefficients to fit a linear model predicting SLA from SSTs that occur one month and two months earlier</span>
<span class="c1"># the returned coefficient parameters  m,c are such that sla = m*sst + c</span>
<span class="n">correlation</span> <span class="o">=</span> <span class="n">sla</span><span class="o">.</span><span class="n">lagged_regression</span><span class="p">(</span><span class="n">sst</span><span class="p">,</span> <span class="n">lags</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">])</span>
</pre></div>
</div>
</dd></dl>

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.timeseries.lagged_regression_month_of_year">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.timeseries.</span></span><span class="sig-name descname"><span class="pre">lagged_regression_month_of_year</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">otherda</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">lags</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">month_of_year</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.timeseries.lagged_regression_month_of_year" title="Permalink to this definition">¶</a></dt>
<dd><p>Obtain linear regression coefficients between this between a yearly timeseries extracted from this DataArray
at a particular month and those from another DataArray, with a series of lags applied, and return the
regression coefficients.</p>
<p>The other DataArray is treated as the y variable, this DataArray is treated as the x variable and the coefficients
returned are the values [m,c] from y = mx+c</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound, assumed to include a “time” dimension</p></li>
<li><p><strong>otherda</strong> (<em>xarray.DataArray</em>) – the other DataArray against which the correlation is to be performed, assumed to have dimensions (time)</p></li>
<li><p><strong>lags</strong> (<em>list</em><em>[</em><em>int</em><em>]</em>) – a list of lags to apply to the time dimension before calculating the regression coefficients
where lag=L means that we compute the regression coefficients for x=self[L:,…] and y=otherda[:-L,…]</p></li>
<li><p><strong>month_of_year</strong> (<em>int</em>) – indicate which month (1=jan, 2=feb, etc) to analyse in the main DataArray</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
dimension and an extra parameter dimension added (with size 2, where index 0 holds the slope value m and index 1
holds the intercept value c)</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
<dt class="field-even">Raises</dt>
<dd class="field-even"><ul class="simple">
<li><p><strong>MisalignedTimeAxisException</strong> – if this array and the other array do not have identical time coordinates</p></li>
<li><p><strong>InvalidMonthOfYearException</strong> – if the month_of_year parameter is not an integer in the range 1-12</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
is imported.</p>
<p class="rubric">Example</p>
<p>A way you might use me is:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">sla</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SLA</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>
<span class="n">sst</span> <span class="o">=</span> <span class="o">...</span> <span class="n">get</span> <span class="n">the</span> <span class="n">monthly</span> <span class="n">SST</span> <span class="k">as</span> <span class="n">a</span> <span class="n">DataArray</span>

<span class="c1"># get the regression coefficients for linear model relating June SLA from May and April SSTs</span>
<span class="c1"># the returned coefficient parameters  m,c are such that sst = m*sla + c</span>
<span class="n">correlation</span> <span class="o">=</span> <span class="n">sla</span><span class="o">.</span><span class="n">lagged_regression_month_of_year</span><span class="p">(</span><span class="n">sst</span><span class="p">,</span> <span class="n">lags</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">],</span> <span class="n">month_of_year</span><span class="o">=</span><span class="mi">6</span><span class="p">)</span>
</pre></div>
</div>
</dd></dl>

</section>
<section id="dataarray-methods-data-reduction">
<h1>DataArray methods - data reduction<a class="headerlink" href="#dataarray-methods-data-reduction" title="Permalink to this headline">¶</a></h1>
<p>include <cite>import xarray_extensions.data_reduction</cite> in your code to add these methods to xarray.DataArray</p>
<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.data_reduction.pca">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.data_reduction.</span></span><span class="sig-name descname"><span class="pre">pca</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">dimension</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">'time'</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">n_components</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">2</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">model_callback</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.data_reduction.pca" title="Permalink to this definition">¶</a></dt>
<dd><p>Extract and return the principle components for this data array after reducing along a target dimension</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound</p></li>
<li><p><strong>dimension</strong> (<em>str</em>) – the name of the target dimension to be reduced</p></li>
<li><p><strong>n_components</strong> (<em>int</em>) – the number of components to extract, which must be less than the length of the target dimension</p></li>
<li><p><strong>model_callback</strong> (<em>function</em>) – if provided, this method will be invoked with a sklearn.decomposition.PCA object after fitting
(see <a class="reference external" href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html</a>)</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance with the target dimension replaced by a dimension of a size
controlled by the n_components parameter</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module is imported</p>
</dd></dl>

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.data_reduction.som">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.data_reduction.</span></span><span class="sig-name descname"><span class="pre">som</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">dimension</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">'time'</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">gridwidth</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">16</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">gridheight</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">16</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">iters</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">25</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">initial_neighbourhood</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">3</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">seed</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">1</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">verbose</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">model_callback</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.data_reduction.som" title="Permalink to this definition">¶</a></dt>
<dd><p>Extract and return the self organising map coordinates for this data array after reducing along a target dimension</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound</p></li>
<li><p><strong>dimension</strong> (<em>str</em>) – the name of the target dimension to be reduced</p></li>
<li><p><strong>iters</strong> (<em>int</em>) – the number of training iterations to use when training the SOM</p></li>
<li><p><strong>gridwidth</strong> (<em>int</em>) – number of cells across the grid</p></li>
<li><p><strong>gridheight</strong> (<em>int</em>) – number of cells down the grid</p></li>
<li><p><strong>initial_neighbourhood</strong> (<em>int</em>) – the initial neighbourhood size</p></li>
<li><p><strong>seed</strong> (<em>int</em>) – random seed - set to produce repeatable results</p></li>
<li><p><strong>verbose</strong> (<em>bool</em>) – whether to print progress messages (SOM can take some time to run)</p></li>
<li><p><strong>model_callback</strong> (<em>function</em>) – if provided, this method will be invoked with a SelfOrganisingMap object
(see source code of this module)</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>an xarray.DataArray instance with the target dimension replaced by a dimension of size 2</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.DataArray</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>This function is attached to the DataArray class as a method when this module is imported</p>
</dd></dl>

</section>
<section id="dataarray-methods-plots">
<h1>DataArray methods - plots<a class="headerlink" href="#dataarray-methods-plots" title="Permalink to this headline">¶</a></h1>
<p>include <cite>import xarray_extensions.plots</cite> in your code to add these methods to xarray.DataArray</p>
<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.plots.rgb_plot">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.plots.</span></span><span class="sig-name descname"><span class="pre">rgb_plot</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">dimension</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">rgb_dimensions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">[0,</span> <span class="pre">1,</span> <span class="pre">None]</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fixed_rgb</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">(0,</span> <span class="pre">0,</span> <span class="pre">0)</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">null_rgb</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">(0.5,</span> <span class="pre">0.5,</span> <span class="pre">0.5)</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.plots.rgb_plot" title="Permalink to this definition">¶</a></dt>
<dd><p>Plot a 3D array relying on a selected dimension of size 1, 2 or 3, for example, an array returned from the
data reduction operators som or pca</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.DataArray</em>) – the DataArray instance to which this method is bound</p></li>
<li><p><strong>dimension</strong> (<em>str</em>) – the name of the target dimension used to derive colour</p></li>
<li><p><strong>rgb_dimensions</strong> (<em>tuple</em><em>[</em><em>int</em><em>,</em><em>int</em><em>,</em><em>int</em><em>]</em>) – indexes into the dimension to use for colours red,green abd blue (or None to use a fixed colour)</p></li>
<li><p><strong>fixed_rgb</strong> (<em>tuple</em><em>(</em><em>float</em><em>,</em><em>float</em><em>,</em><em>float</em><em>)</em>) – for colour dimensions which are not indexed (rgb_dimensions parameter), specify the value in the range 0-1</p></li>
<li><p><strong>null_rgb</strong> (<em>tuple</em><em>(</em><em>float</em><em>,</em><em>float</em><em>,</em><em>float</em><em>)</em>) – plot nan/null values using this rgb colour (values in the range 0-1)</p></li>
</ul>
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

<dl class="py function">
<dt class="sig sig-object py" id="xarray_extensions.general.longitude_center_zero">
<span class="sig-prename descclassname"><span class="pre">xarray_extensions.general.</span></span><span class="sig-name descname"><span class="pre">longitude_center_zero</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">self</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">longitude_name</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">'lon'</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#xarray_extensions.general.longitude_center_zero" title="Permalink to this definition">¶</a></dt>
<dd><p>Center longitude axis on 0.  If the original longitude axis ranges from 0 &lt;= lon &lt; 360, modify the dataset so that the new one should range from -180 to 180.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>self</strong> (<em>xarray.Dataset</em>) – the DataSet instance to be modified in place</p></li>
<li><p><strong>longitude_name</strong> (<em>str</em>) – the name of the longitude dimension</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p>A new dataset with the longitude axis adjusted and data arrays shifted accordingly.  The original attributes of the
longitude dimension are retained, but valid_min and valid_max are modified (if present)</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>xarray.Dataset</p>
</dd>
<dt class="field-even">Raises</dt>
<dd class="field-even"><p><strong>Exception</strong> – If the input longitude axis has values outside the range 0 &lt;= lon &lt; 360</p>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>Contributed by Ross Maidment</p>
</dd></dl>

</section>
