The Tucker manifold \(\mathcal{M}_{\mathbf{r}}\) consists of all order-\(d\) tensors in \(\mathbb{R}^{n_1 \times \cdots \times n_d}\) with fixed multilinear rank \(\mathbf{r} = (r_1, \dots, r_d)\), where \(r_k \leq n_k\) for each mode \(k\). A point \(X \in \mathcal{M}_{\mathbf{r}}\) admits a Tucker decomposition \(X = G \times_1 U_1 \times_2 \cdots \times_d U_d\), where \(G \in \mathbb{R}^{r_1 \times \cdots \times r_d}\) is the core tensor (assumed to have full multilinear rank) and each \(U_k \in \mathbb{R}^{n_k \times r_k}\) is a column-orthogonal factor matrix (i.e., \(U_k^T U_k = I_{r_k}\)).

The tangent space \(T_X \mathcal{M}_{\mathbf{r}}\) at \(X\) parametrizes infinitesimal variations that preserve the manifold structure. It can be expressed as
\[
T_X \mathcal{M}_{\mathbf{r}} = \left\{ \delta G \times_1 U_1 \times_2 \cdots \times_d U_d + \sum_{k=1}^d G \times_k \delta U_k \times_{j \neq k} U_j \;\middle|\; \delta G \in \mathbb{R}^{r_1 \times \cdots \times r_d},\ \delta U_k \in \mathbb{R}^{n_k \times r_k},\ \delta U_k^T U_k = 0 \right\},
\]
where the condition \(\delta U_k^T U_k = 0\) ensures the variations \(\delta U_k\) are orthogonal to the span of \(U_k\), maintaining the Stiefel manifold constraints on the factors.

An alternative formulation expresses the tangent space using orthogonal complements \(U_k^\perp \in \mathbb{R}^{n_k \times (n_k - r_k)}\) (with \((U_k^\perp)^T U_k^\perp = I_{n_k - r_k}\) and \((U_k^\perp)^T U_k = 0\)):
\[
T_X \mathcal{M}_{\mathbf{r}} = \left\{ \dot{G} \times_1 U_1 \times_2 \cdots \times_d U_d + \sum_{k=1}^d (G \times_k \dot{R}_k) \times_k U_k^\perp \times_{j \neq k} U_j \;\middle|\; \dot{G} \in \mathbb{R}^{r_1 \times \cdots \times r_d},\ \dot{R}_k \in \mathbb{R}^{(n_k - r_k) \times r_k} \right\}.
\]
This parametrization avoids the explicit orthogonality constraint by directly using a basis for the orthogonal complement in each mode, which can be computationally advantageous.

The orthogonal projection of an arbitrary tensor \(A \in \mathbb{R}^{n_1 \times \cdots \times n_d}\) onto \(T_X \mathcal{M}_{\mathbf{r}}\) (with respect to the Frobenius inner product) decomposes into a "core variation" term plus "mode variation" terms:
\[
P_{T_X \mathcal{M}_{\mathbf{r}}}(A) = A \times_{k=1}^d P_{U_k} + \sum_{k=1}^d G \times_k \left( P_{\perp U_k} \left( A \times_{j \neq k} U_j^T \right)^{(k)} G^{(k)\dagger} \right) \times_{j \neq k} U_j,
\]
where:
- \(P_{U_k} = U_k U_k^T\) is the orthogonal projector onto the column span of \(U_k\),
- \(P_{\perp U_k} = I_{n_k} - P_{U_k}\) is the projector onto its orthogonal complement,
- \(A^{(k)}\) and \(G^{(k)}\) denote the mode-\(k\) matrix unfoldings of \(A\) and \(G\), respectively,
- \(G^{(k)\dagger}\) is the Moore-Penrose pseudoinverse of \(G^{(k)}\) (which simplifies to \(G^{(k)T} (G^{(k)} G^{(k)T})^{-1}\) if \(G^{(k)}\) has full row rank, as is typical in the interior of the manifold),
- \(\times_k\) denotes mode-\(k\) multiplication, and \(\times_{j \neq k} U_j\) is shorthand for multiplying by \(U_j\) in all modes except \(k\).

To derive this formula, start from the parametrization of \(T_X \mathcal{M}_{\mathbf{r}}\) and solve for the unique \(\delta G, \delta U_k\) (satisfying the constraints) that minimize \(\|A - \delta X\|_F^2\), where \(\delta X \in T_X \mathcal{M}_{\mathbf{r}}\). This leads to a system where the core variation is isolated via projections onto the current subspaces (\(A \times_{k=1}^d P_{U_k}\)), and each mode variation is recovered by unfolding, projecting orthogonally, and "inverting" through the core's pseudoinverse before refolding and remultiplying by the factors.

This projection arises in applications like Riemannian optimization on \(\mathcal{M}_{\mathbf{r}}\) (e.g., for low-rank tensor completion) and dynamical low-rank approximation, where the right-hand side of an evolution equation is projected onto the tangent space to evolve the approximation while preserving rank.