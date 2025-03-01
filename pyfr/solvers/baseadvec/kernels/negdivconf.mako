<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
% for mod, name in src_macros:
    <%include file='${mod}'/>
% endfor

<%pyfr:kernel name='negdivconf' ndim='2'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
fpdtype_t src[${nvars}] = {};

% for mod, name in src_macros:
    ${pyfr.expand(name, 't', 'u', 'ploc', 'src')};
% endfor

% for i in range(nvars):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}];
% endfor

tdivtconf[1] += src[0]*src[0]*u[0]*ploc[0] + 2*src[0]*u[2] + u[0]*src[1]*ploc[1];
tdivtconf[2] += src[0]*src[0]*u[0]*ploc[1] - 2*src[0]*u[1] - u[0]*src[1]*ploc[0];
</%pyfr:kernel>
