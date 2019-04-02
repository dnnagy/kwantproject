Computing local quantities: densities and currents
==================================================
In the previous tutorials we have mainly concentrated on calculating *global*
properties such as conductance and band structures. Often, however, insight can
be gained from calculating *locally-defined* quantities, that is, quantities
defined over individual sites or hoppings in your system. In the
:ref:`closed-systems` tutorial we saw how we could visualize the density
associated with the eigenstates of a system using `kwant.plotter.map`.

In this tutorial we will see how we can calculate more general quantities than
simple densities by studying spin transport in a system with a magnetic
texture.

.. seealso::
    The complete source code of this example can be found in
    :download:`magnetic_texture.py </code/download/magnetic_texture.py>`


Introduction
------------
Our starting point will be the following spinful tight-binding model on
a square lattice:

.. math::
    H = - \sum_{⟨ij⟩}\sum_{α} |iα⟩⟨jα|
        + J \sum_{i}\sum_{αβ} \mathbf{m}_i⋅ \mathbf{σ}_{αβ} |iα⟩⟨iβ|,

where latin indices run over sites, and greek indices run over spin.  We can
identify the first term as a nearest-neighbor hopping between like-spins, and
the second as a term that couples spins on the same site.  The second term acts
like a magnetic field of strength :math:`J` that varies from site to site and
that, on site :math:`i`, points in the direction of the unit vector
:math:`\mathbf{m}_i`. :math:`\mathbf{σ}_{αβ}` is a vector of Pauli matrices.
We shall take the following form for :math:`\mathbf{m}_i`:

.. math::
    \mathbf{m}_i &=\ \left(
        \frac{x_i}{x_i^2 + y_i^2} \sin θ_i,\
        \frac{y_i}{x_i^2 + y_i^2} \sin θ_i,\
        \cos θ_i \right)^T,
    \\
    θ_i &=\ \frac{π}{2} (\tanh \frac{r_i - r_0}{δ} - 1),

where :math:`x_i` and :math:`y_i` are the :math:`x` and :math:`y` coordinates
of site :math:`i`, and :math:`r_i = \sqrt{x_i^2 + y_i^2}`.

To define this model in Kwant we start as usual by defining functions that
depend on the model parameters:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_model
    :end-before: #HIDDEN_END_model

and define our system as a square shape on a square lattice with two orbitals
per site, with leads attached on the left and right:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_syst
    :end-before: #HIDDEN_END_syst

Below is a plot of a projection of :math:`\mathbf{m}_i` onto the x-y plane
inside the scattering region. The z component is shown by the color scale:

.. image:: /code/figure/mag_field_direction.*

We will now be interested in analyzing the form of the scattering states
that originate from the left lead:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_wavefunction
    :end-before: #HIDDEN_END_wavefunction

Local densities
---------------
If we were simulating a spinless system with only a single degree of freedom,
then calculating the density on each site would be as simple as calculating the
absolute square of the wavefunction like::

    density = np.abs(psi)**2

When there are multiple degrees of freedom per site, however, one has to be
more careful. In the present case with two (spin) degrees of freedom per site
one could calculate the per-site density like:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_ldos
    :end-before: #HIDDEN_END_ldos

With more than one degree of freedom per site we have more freedom as to what
local quantities we can meaningfully compute. For example, we may wish to
calculate the local z-projected spin density. We could calculate
this in the following way:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_lsdz
    :end-before: #HIDDEN_END_lsdz

If we wanted instead to calculate the local y-projected spin density, we would
need to use an even more complicated expression:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_lsdy
    :end-before: #HIDDEN_END_lsdy

The `kwant.operator` module aims to alleviate somewhat this tedious
book-keeping by providing a simple interface for defining operators that act on
wavefunctions. To calculate the above quantities we would use the
`~kwant.operator.Density` operator like so:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_lden
    :end-before: #HIDDEN_END_lden

`~kwant.operator.Density` takes a `~kwant.system.System` as its first parameter
as well as (optionally) a square matrix that defines the quantity that you wish
to calculate per site. When an instance of a `~kwant.operator.Density` is then
evaluated with a wavefunction, the quantity

.. math:: ρ_i = \mathbf{ψ}^†_i \mathbf{M} \mathbf{ψ}_i

is calculated for each site :math:`i`, where :math:`\mathbf{ψ}_{i}` is a vector
consisting of the wavefunction components on that site and :math:`\mathbf{M}`
is the square matrix referred to previously.

Below we can see colorplots of the above-calculated quantities. The array that
is returned by evaluating a `~kwant.operator.Density` can be used directly with
`kwant.plotter.density`:

.. image:: /code/figure/spin_densities.*


.. specialnote:: Technical Details

    Although we refer loosely to "densities" and "operators" above, a
    `~kwant.operator.Density` actually represents a *collection* of linear
    operators. This can be made clear by rewriting the above definition
    of :math:`ρ_i` in the following way:

    .. math::
        ρ_i = \sum_{αβ} ψ^*_{α} \mathcal{M}_{iαβ} ψ_{β}

    where greek indices run over the degrees of freedom in the Hilbert space of
    the scattering region and latin indices run over sites.  We can this
    identify :math:`\mathcal{M}_{iαβ}` as the components of a rank-3 tensor and can
    represent them as a "vector of matrices":

    .. math::
        \mathcal{M} = \left[
        \left(\begin{matrix}
            \mathbf{M} & 0 & … \\
            0 & 0 & … \\
            ⋮ & ⋮ & ⋱
        \end{matrix}\right)
        ,\
        \left(\begin{matrix}
            0 & 0 & … \\
            0 & \mathbf{M} & … \\
            ⋮ & ⋮ & ⋱
        \end{matrix}\right)
        , … \right]

    where :math:`\mathbf{M}` is defined as in the main text, and the :math:`0`
    are zero matrices of the same shape as :math:`\mathbf{M}`.


Local currents
--------------
`kwant.operator` also has a class `~kwant.operator.Current` for calculating
local currents, analogously to the local "densities" described above. If
one has defined a density via a matrix :math:`\mathbf{M}` and the above
equation, then one can define a local current flowing from site :math:`b`
to site :math:`a`:

.. math:: J_{ab} = i \left(
    \mathbf{ψ}^†_b (\mathbf{H}_{ab})^† \mathbf{M} \mathbf{ψ}_a
    - \mathbf{ψ}^†_a \mathbf{M} \mathbf{H}_{ab} \mathbf{ψ}_b
    \right),

where :math:`\mathbf{H}_{ab}` is the hopping matrix from site :math:`b` to site
:math:`a`.  For example, to calculate the local current and
spin current:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_current
    :end-before: #HIDDEN_END_current

Evaluating a `~kwant.operator.Current` operator on a wavefunction returns a
1D array of values that can be directly used with `kwant.plotter.current`:

.. image:: /code/figure/spin_currents.*

.. note::

    Evaluating a `~kwant.operator.Current` operator on a wavefunction
    returns a 1D array of the same length as the number of hoppings in the
    system, ordered in the same way as the edges in the system's graph.

.. specialnote:: Technical Details

    Similarly to how we saw in the previous section that `~kwant.operator.Density`
    can be thought of as a collection of operators, `~kwant.operator.Current`
    can be defined in a similar way. Starting from the definition of a "density":

    .. math:: ρ_a = \sum_{αβ} ψ^*_{α} \mathcal{M}_{aαβ} ψ_{β},

    we can define *currents* :math:`J_{ab}` via the continuity equation:

    .. math:: \frac{∂ρ_a}{∂t} - \sum_{b} J_{ab} = 0

    where the sum runs over sites :math:`b` neigboring site :math:`a`.
    Plugging in the definition for :math:`ρ_a`, along with the Schrödinger
    equation and the assumption that :math:`\mathcal{M}` is time independent,
    gives:

    .. math:: J_{ab} = \sum_{αβ}
        ψ^*_α \left(i \sum_{γ}
            \mathcal{H}^*_{abγα} \mathcal{M}_{aγβ}
            - \mathcal{M}_{aαγ} \mathcal{H}_{abγβ}
        \right)  ψ_β,

    where latin indices run over sites and greek indices run over the Hilbert
    space degrees of freedom, and

    .. math:: \mathcal{H}_{ab} = \left(\begin{matrix}
            ⋱ & ⋮ & ⋮ & ⋮ & ⋰ \\
            ⋯ & ⋱ & 0 & \mathbf{H}_{ab} & ⋯ \\
            ⋯ & 0 & ⋱ & 0 & ⋯ \\
            ⋯ & 0 & 0 & ⋱ & ⋯ \\
            ⋰ & ⋮ & ⋮ & ⋮ & ⋱
        \end{matrix}\right).

    i.e. :math:`\mathcal{H}_{ab}` is a matrix that is zero everywhere
    except on elements connecting *from* site :math:`b` *to* site :math:`a`,
    where it is equal to the hopping matrix :math:`\mathbf{H}_{ab}` between
    these two sites.

    This allows us to identify the rank-4 quantity

    .. math:: \mathcal{J}_{abαβ} = i \sum_{γ}
            \mathcal{H}^*_{abγα} \mathcal{M}_{aγβ}
            - \mathcal{M}_{aαγ} \mathcal{H}_{abγβ}

    as the local current between connected sites.

    The diagonal part of this quantity, :math:`\mathcal{J}_{aa}`,
    represents the extent to which the density defined by :math:`\mathcal{M}_a`
    is not conserved on site :math:`a`. It can be calculated using
    `~kwant.operator.Source`, rather than `~kwant.operator.Current`, which
    only computes the off-diagonal part.


Spatially varying operators
---------------------------
The above examples are reasonably simple in the sense that the book-keeping
required to manually calculate the various densities and currents is still
manageable. Now we shall look at the case where we wish to calculate some
projected spin currents, but where the spin projection axis varies from place
to place. More specifically, we want to visualize the spin current along the
direction of :math:`\mathbf{m}_i`, which changes continuously over the whole
scattering region.

Doing this is as simple as passing a *function* when instantiating
the `~kwant.operator.Current`, instead of a constant matrix:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_following
    :end-before: #HIDDEN_END_following

The function must take a `~kwant.builder.Site` as its first parameter,
and may optionally take other parameters (i.e. it must have the same
signature as a Hamiltonian onsite function), and must return the square
matrix that defines the operator we wish to calculate.

.. note::

    In the above example we had to pass the extra parameters needed by the
    ``following_operator`` function via the ``param`` keyword argument.  In
    general you must pass all the parameters needed by the Hamiltonian via
    ``params`` (as you would when calling `~kwant.solvers.default.smatrix` or
    `~kwant.solvers.default.wave_function`).  In the previous examples,
    however, we used the fact that the system hoppings do not depend on any
    parameters (these are the only Hamiltonian elements required to calculate
    currents) to avoid passing the system parameters for the sake of brevity.

Using this we can see that the spin current is essentially oriented along
the direction of :math:`m_i` in the present regime where the onsite term
in the Hamiltonian is dominant:

.. image:: /code/figure/spin_current_comparison.*

.. note:: Although this example used exclusively `~kwant.operator.Current`,
          you can do the same with `~kwant.operator.Density`.


Defining operators over parts of a system
-----------------------------------------

Another useful feature of `kwant.operator` is the ability to calculate
operators over selected parts of a system. For example, we may wish to
calculate the total density of states in a certain part
of the system, or the current flowing through a cut in the system.
We can do this selection when creating the operator by using the
keyword parameter ``where``.

Density of states in a circle
*****************************

To calculate the density of states inside a circle of radius
20 we can simply do:

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_density_cut
    :end-before: #HIDDEN_END_density_cut

.. literalinclude:: /code/figure/circle_dos.txt

note that we also provide ``sum=True``, which means that evaluating the
operator on a wavefunction will produce a single scalar. This is semantically
equivalent to providing ``sum=False`` (the default) and running ``numpy.sum``
on the output.

Current flowing through a cut
*****************************

Below we calculate the probability current and z-projected spin current near
the interfaces with the left and right leads.

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_current_cut
    :end-before: #HIDDEN_END_current_cut

.. literalinclude:: /code/figure/current_cut.txt

We see that the probability current is conserved across the scattering region,
but the z-projected spin current is not due to the fact that the Hamiltonian
does not commute with :math:`σ_z` everywhere in the scattering region.

.. note:: ``where`` can also be provided as a sequence of `~kwant.builder.Site`
          or a sequence of hoppings (i.e. pairs of `~kwant.builder.Site`),
          rather than a function.


Advanced Topics
---------------

Using ``bind`` for speed
************************
In most of the above examples we only used each operator *once* after creating
it. Often one will want to evaluate an operator with many different
wavefunctions, for example with all scattering wavefunctions at a certain
energy, but with the *same set of parameters*. In such cases it is best to tell
the operator to pre-compute the onsite matrices and any necessary Hamiltonian
elements using the given set of parameters, so that this work is not duplicated
every time the operator is evaluated.

This can be achieved with `~kwant.operator.Current.bind`:

.. warning:: Take care that you do not use an operator that was bound to a
             particular set of parameters with wavefunctions calculated with a
             *different* set of parameters. This will almost certainly give
             incorrect results.

.. literalinclude:: /code/include/magnetic_texture.py
    :start-after: #HIDDEN_BEGIN_bind
    :end-before: #HIDDEN_END_bind

.. image:: /code/figure/bound_current.*
