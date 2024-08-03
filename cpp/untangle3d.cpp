#include <iostream>
#include <fstream>
#include <limits>
#include <cassert>
#include <cstring>
#include <chrono>

#include <ultimaille/all.h>

#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>



using namespace UM;

inline double chi(double eps, double det) {
    if (det>0)
        return (det + std::sqrt(eps*eps + det*det))*.5;
    return .5*eps*eps / (std::sqrt(eps*eps + det*det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5+det/(2.*std::sqrt(eps*eps + det*det));
}

struct Untangle3D {
    Untangle3D(Tetrahedra &mesh) : m(mesh), X(m.nverts()*3), lock(m.points, false), J(m), K(m), det(m), ref_tet(m), volume(m) {
        for (int t : cell_iter(m)) {
            volume[t] = m.util.cell_volume(t);
#if 1
            mat<3,3> ST = {{
                m.points[m.vert(t, 1)] - m.points[m.vert(t, 0)],
                m.points[m.vert(t, 2)] - m.points[m.vert(t, 0)],
                m.points[m.vert(t, 3)] - m.points[m.vert(t, 0)]
            }};
#else
            Tetrahedra R; // regular tetrahedron with unit edge length, centered at the origin (sqrt(2)/12 volume)
            R.cells = {0,1,2,3};
            *R.points.data = {
                { .5,   0, -1./(2.*std::sqrt(2.))},
                {-.5,   0, -1./(2.*std::sqrt(2.))},
                {  0, -.5,  1./(2.*std::sqrt(2.))},
                {  0,  .5,  1./(2.*std::sqrt(2.))}
            };
            double a = std::cbrt(volume[t]*6.*std::sqrt(2.));
            for (vec3 &p : R.points) // scale the tet
                p = p*a;
            mat<3,3> ST = {{
                R.points[1] - R.points[0],
                R.points[2] - R.points[0],
                R.points[3] - R.points[0]
            }};
#endif
            ref_tet[t] = mat<4,3>{{ {-1,-1,-1},{1,0,0},{0,1,0},{0,0,1} }}*ST.invert_transpose();
        }
    }

    void lock_boundary_verts() {
        VolumeConnectivity vec(m);
        for (int c : cell_iter(m))
            for (int lf : range(4))
                if (vec.adjacent[m.facet(c, lf)]<0)
                    for (int lv : range(3))
                        lock[m.facet_vert(c, lf, lv)] = true;
    }

    void evaluate_jacobian(const std::vector<double> &X) {
        detmin = std::numeric_limits<double>::max();
        ninverted = 0;
#pragma omp parallel for reduction(min:detmin) reduction(+:ninverted)
        for (int c=0; c<m.ncells(); c++) {
            mat<3,3> &J = this->J[c];
            J = {};
            for (int i=0; i<4; i++)
                for (int d : range(3))
                    J[d] += ref_tet[c][i]*X[3*m.vert(c,i) + d];
            det[c] = J.det();
            detmin = std::min(detmin, det[c]);
            ninverted += (det[c]<=0);

            this->K[c] = { // dual basis
                {{
                     J[1].y*J[2].z - J[1].z*J[2].y,
                     J[1].z*J[2].x - J[1].x*J[2].z,
                     J[1].x*J[2].y - J[1].y*J[2].x
                 },
                {
                    J[0].z*J[2].y - J[0].y*J[2].z,
                    J[0].x*J[2].z - J[0].z*J[2].x,
                    J[0].y*J[2].x - J[0].x*J[2].y
                },
                {
                    J[0].y*J[1].z - J[0].z*J[1].y,
                    J[0].z*J[1].x - J[0].x*J[1].z,
                    J[0].x*J[1].y - J[0].y*J[1].x
                }}
            };
        }
    }

    bool go() {
        std::vector<SpinLock> spin_locks(X.size());
        eps = 1.;
        evaluate_jacobian(X);
        if (debug>0) std::cerr <<  "number of inverted elements: " << ninverted << std::endl;
        for (int iter=0; iter<maxiter; iter++) {
            if (debug>0) std::cerr << "iteration #" << iter << std::endl;

            const LBFGS_Optimizer::func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
                std::fill(G.begin(), G.end(), 0);
                F = 0;
                evaluate_jacobian(X);
#pragma omp parallel for reduction(+:F)
                for (int t=0; t<m.ncells(); t++) {
                    mat<3,3> &a = this->J[t]; // tangent basis
                    mat<3,3> &b = this->K[t]; // dual basis
                    double c1 = chi(eps, det[t]);
                    double c2 = pow(c1, 2./3.);
                    double c3 = chi_deriv(eps, det[t]);

                    double f = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2])/c2;
                    double g = (1+det[t]*det[t])/c1;
                    F += ((1-theta)*f + theta*g)*volume[t];

                    for (int dim : range(3)) {
                        vec3 dfda = a[dim]*(2./c2) - b[dim]*((2.*f*c3)/(3.*c1));
                        vec3 dgda = b[dim]*((2*det[t]-g*c3)/c1);

                        for (int i=0; i<4; i++) {
                            int v = m.vert(t,i);
                            if (lock[v]) continue;
                            spin_locks[v*3+dim].lock();
                            G[v*3+dim] += ((dfda*(1.-theta) + dgda*theta)*ref_tet[t][i])*volume[t];
                            spin_locks[v*3+dim].unlock();
                        }
                    }
                }
            };

            double E_prev, E;
            std::vector<double> trash(X.size());
            func(X, E_prev, trash);

            LBFGS_Optimizer opt(func);
            opt.gtol = bfgs_threshold;
            opt.maxiter = bfgs_maxiter;
            opt.run(X);

            func(X, E, trash);
            if (debug>0) std::cerr << "E: " << E << " eps: " << eps << " detmin: " << detmin << " ninv: " << ninverted << std::endl;

            double sigma = std::max(1.-E/E_prev, 1e-1);
            double mu = (1-sigma)*chi(eps, detmin);
            if (detmin<mu)
                eps = 2*std::sqrt(mu*(mu-detmin));
            else eps = 1e-10;

            if  (detmin>0 && std::abs(E_prev - E)/E<1e-5) break;
        }
        return !ninverted;
    }

    ////////////////////////////////
    // Untangle3D state variables //
    ////////////////////////////////

    // optimization input parameters
    Tetrahedra &m;          // the mesh to optimize
    double theta = 1./2.;   // the energy is (1-theta)*(shape energy) + theta*(area energy)
    int maxiter = 10000;    // max number of outer iterations
    int bfgs_maxiter = 3000; // max number of inner iterations
    double bfgs_threshold = 1e-4;

    int debug = 1;          // verbose level

    // optimization state variables

    std::vector<double> X;     // current geometry
    PointAttribute<bool> lock; // currently lock = boundary vertices
    CellAttribute<mat<3,3>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    CellAttribute<mat<3,3>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    CellAttribute<double> det; // per-tet determinant of the Jacobian matrix
    CellAttribute<mat<4,3>> ref_tet;   // reference tetrahedron: array of 4 normal vectors to compute the gradients
    CellAttribute<double> volume; // reference volume
    double eps;       // regularization parameter, depends on min(jacobian)

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra
};



//---------------------------------------------------------------------------------- CGAL stuff
using CGAL_Kernel      = CGAL::Exact_predicates_inexact_constructions_kernel;
using CGAL_Point3      = CGAL::Point_3<CGAL_Kernel>;
using CGAL_Tetrahedron = CGAL::Tetrahedron_3<CGAL_Kernel>;

CGAL_Point3 to_cgal(const vec3 p){
    return {p.x, p.y, p.z};
}



bool extract_mesh_name_and_directory(std::string location,
                                     std::string& filename,
                                     std::string& directory){

    char sep1 = '/';

    size_t directory_end_index = location.rfind(sep1, location.length());

    if (directory_end_index != std::string::npos) {
        //-4 to get rid of extension
        filename = location.substr(directory_end_index+1, location.length()-directory_end_index-5);
        directory = location.substr(0, directory_end_index);

        return true;
    }

    return false;
}

void compute_volume(Tetrahedra& mesh, double& um_vol, double& cgal_vol){
    um_vol = 0;
    cgal_vol = 0;
    for (int t : cell_iter(mesh)) {
        um_vol += mesh.util.cell_volume(t);

        auto v0 = mesh.points[mesh.vert(t, 0)];
        auto v1 = mesh.points[mesh.vert(t, 1)];
        auto v2 = mesh.points[mesh.vert(t, 2)];
        auto v3 = mesh.points[mesh.vert(t, 3)];

        CGAL_Tetrahedron cgal_tet(to_cgal(v0),
                                  to_cgal(v1),
                                  to_cgal(v2),
                                  to_cgal(v3));

        cgal_vol += cgal_tet.volume();
    }
}


void scale_to_volume(Tetrahedra& mesh, double vol = 1.0){
    double um_vol, cgal_vol(0);
    compute_volume(mesh, um_vol, cgal_vol);


    std::cout<<" - initial volume = "<<cgal_vol<<std::endl;


    for (vec3 &p : mesh.points)
        p = p * std::pow(vol / cgal_vol,  1./3.);

    compute_volume(mesh, um_vol, cgal_vol);
    std::cout<<" - scaled volume = "<<cgal_vol<<std::endl;
}


bool check_validity_with_cgal(const Tetrahedra& mesh, const std::string& res_filename){

    int neg_count(0);
    int deg_count(0);
    int pos_count(0);

    double um_volume(0);
    double cgal_volume(0);

    for (int t : cell_iter(mesh)) {
        um_volume += mesh.util.cell_volume(t);

        auto v0 = mesh.points[mesh.vert(t, 0)];
        auto v1 = mesh.points[mesh.vert(t, 1)];
        auto v2 = mesh.points[mesh.vert(t, 2)];
        auto v3 = mesh.points[mesh.vert(t, 3)];

        CGAL_Tetrahedron cgal_tet(to_cgal(v0),
                                  to_cgal(v1),
                                  to_cgal(v2),
                                  to_cgal(v3));

        cgal_volume += cgal_tet.volume();

        auto orientation = cgal_tet.orientation();
        neg_count += orientation == CGAL::NEGATIVE;
        deg_count += orientation == CGAL::ZERO;
        pos_count += orientation == CGAL::POSITIVE;

        /*m.points[m.vert(t, 1)] - m.points[m.vert(t, 0)],
         m.points[m.vert(t, 2)] - m.points[m.vert(t, 0)],
         m.points[m.vert(t, 3)] - m.points[m.vert(t, 0)]*/

    }


    /*for(int i(0); i<mesh.points.size(); i++){
        std::cout<<" - "<<i<<" at "<<std::setprecision(64)<<mesh.points[i]<<std::endl;
    }*/

    std::cout<<" - check: mesh volume from ultimaille = "<<um_volume<<std::endl;
    std::cout<<" - check: mesh volume from       CGAL = "<<cgal_volume<<std::endl;

    std::cout<<" -     flipped count: "<<neg_count<<std::endl;
    std::cout<<" -  degenerate count: "<<deg_count<<std::endl;
    std::cout<<" - non-flipped count: "<<pos_count<<std::endl;

    std::string filename, super_directory, directory;

    if(!extract_mesh_name_and_directory(res_filename,
                                        filename,
                                        directory)){
            std::cout<<" ERROR - couldn't extract filename and location"<<std::endl;
    }else{
        if(!extract_mesh_name_and_directory(directory,
                                            filename,
                                            super_directory)){
            std::cout<<" ERROR - couldn't extract super filename and location"<<std::endl;
        }else{


            std::string output_path = directory+"/"+filename+"_FOF_stats.txt";
            std::cout<<" -> writing stats to "<<output_path<<std::endl;

            std::ofstream output_file(output_path);
            if(!output_file.is_open()){
                std::cout<<" ERROR - couldn't open stat file "<<output_path<<std::endl;
                return -1;
            }

            std::cout<<" res_filename: "<<res_filename<<std::endl;
            std::cout<<" filename: "<<filename<<std::endl;
            std::cout<<" directory: "<<directory<<std::endl;
            std::cout<<" super_directory: "<<super_directory<<std::endl;

            int nedges(0);
            output_file<<filename<<std::endl;
            output_file<<mesh.nverts()<<std::endl;
            output_file<<nedges<<std::endl;
            output_file<<mesh.nfacets()<<std::endl;
            output_file<<mesh.ncells()<<std::endl;
            output_file<<deg_count<<std::endl;
            output_file<<neg_count<<std::endl;
            output_file<<0<<std::endl;

            output_file.close();
        }
    }

    return !neg_count && !deg_count;
}

//---------------------------------------------------------------------------- end of CGAL stuff


int main(int argc, char** argv) {
    if (3>argc) {
        std::cerr << "Usage: " << argv[0] << " init.mesh reference.mesh [result.mesh]" << std::endl;
        return 1;
    }

    std::string res_filename = "result.mesh";
    if (4<=argc) {
        res_filename = std::string(argv[3]);
    }

    Tetrahedra ini, ref;
    read_by_extension(argv[1], ini);
    read_by_extension(argv[2], ref);
    std::cerr << "Untangling " << argv[1] << "," << ini.nverts() << "," << std::endl;

    if (ini.nverts()!=ref.nverts() || ini.ncells()!=ref.ncells()) {
        std::cerr << "Error: " << argv[1] << " and " << argv[2] << " must have the same number of vertices and tetrahedra, aborting" << std::endl;
        return -1;
    }

/*
    std::vector<bool> tokill(ref.ncells(), false);
    std::vector<std::array<int, 4>> new_cells;
    VolumeConnectivity vec(ref);
    for (int c : cell_iter(ref)) {
        int cf2 = -1;
        int cf1 = -1;
        for (int cf : range(4)) if (vec.adjacent[4*c + cf] == -1) {
            if (cf1 == -1) cf1 = cf;
            else cf2 = cf;
        }
        if (cf2<0) continue;

        int he = vec.halfedge(c, cf1, 0);
        for (int i : range(2)) if (vec.cell_facet(vec.opposite_f(he)) != cf2) he = vec.next(he);

        he = vec.prev(vec.opposite_f(vec.next(he)));

        int mid = ref.nverts();
        ref.points.push_back(0.5 * (ref.points[vec.from(he)] + ref.points[vec.to(he)]));
        ini.points.push_back(0.5 * (ini.points[vec.from(he)] + ini.points[vec.to(he)]));

        for (int he2split : vec.halfedges_around_edge(he)) {
            tokill[vec.cell(he2split)] = true;
            new_cells.push_back({ vec.from(he2split), vec.to(vec.next(he2split)), mid, vec.to(vec.next(vec.opposite_f(he2split))) });
            he2split = vec.opposite_f(he2split);
            new_cells.push_back({ vec.from(he2split), vec.to(vec.next(he2split)), mid, vec.to(vec.next(vec.opposite_f(he2split))) });
        }
    }
    ref.delete_cells(tokill);
    ini.delete_cells(tokill);

    {
        int off = ref.create_cells(new_cells.size());
        for (int i : range(new_cells.size())) for (int lv : range(4)) ref.vert(off + i, lv) = new_cells[i][lv];
    }
    {
        int off = ini.create_cells(new_cells.size());
        for (int i : range(new_cells.size())) for (int lv : range(4)) ini.vert(off + i, lv) = new_cells[i][lv];
    }

    write_by_extension("split-rest.mesh", ref);
    write_by_extension("split-init.mesh", ini);
//  return 0;
*/



#if 0
    Permutation perm(ref.nverts());
    Permutation perm2(ref.nverts());
    HilbertSort hs(*ref.points.data);
    hs.apply(perm.ind);
    perm.apply(*ref.points.data);
    perm.apply(*ini.points.data);
    perm.apply_reverse(perm2.ind);
    for (int t : cell_iter(ref))
        for (int lv : range(4))
            ini.vert(t, lv) = ref.vert(t, lv) = perm2[ref.vert(t, lv)];
    write_geogram("gna.geogram", ref);
#endif


    bool inverted = false;
    { // ascertain the mesh requirements
        double ref_volume = 0, ini_volume = 0;
        for (int c : cell_iter(ref)) {
            ref_volume += ref.util.cell_volume(c);
            ini_volume += ini.util.cell_volume(c);

        }

        if (
                (ref_volume<0 && ini_volume>0) ||
                (ref_volume>0 && ini_volume<0)
           ) {
            std::cerr << "Error: " << argv[1] << " and " << argv[2] << " must have the orientation, aborting" << std::endl;
            return -1;
        }

        inverted = (ini_volume<=0);
        if (inverted) {
            std::cerr << "Warning: the input has negative volume, inverting" << std::endl;
            for (vec3 &p : ini.points)
                p.x *= -1;
            for (vec3 &p : ref.points)
                p.x *= -1;
        }
    }

    vec3 bbmin, bbmax; // these are used to undo the scaling we apply to the model
    const double boxsize = 10.;

    /*{ // scale
        ref.points.util.bbox(bbmin, bbmax);
        double maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
        for (vec3 &p : ref.points)
            p = (p - (bbmax+bbmin)/2.)*boxsize/maxside + vec3(1,1,1)*boxsize/2;
        for (vec3 &p : ini.points)
            p = (p - (bbmax+bbmin)/2.)*boxsize/maxside + vec3(1,1,1)*boxsize/2;
    }*/

    double um_vol, cgal_vol;
    //compute_volume(ref, um_vol, cgal_vol);
    //scale_to_volume(ini, 1.0);
    //scale_to_volume(ref, 1.0);


    auto cgal_ini_valid = check_validity_with_cgal(ini, "");
    auto cgal_ref_valid = check_validity_with_cgal(ref, "");
    std::cout<<"   initial mesh valid per CGAL: "<<cgal_ini_valid<<std::endl;
    std::cout<<" reference mesh valid per CGAL: "<<cgal_ref_valid<<std::endl;

    //std::cout<<" stopping here for now"<<std::endl;
    //exit(EXIT_FAILURE);

    Untangle3D opt(ref);

    for (int v : vert_iter(ref)){
        for (int d : range(3)){
            opt.X[3*v+d] = ini.points[v][d];
        }
        if(!v){
            std::cout<<" - codomain v0: "<<opt.X[0]<<" "<<opt.X[1]<<" "<<opt.X[2]<<" "<<std::endl;
        }
    }

    opt.lock_boundary_verts();

    auto t1 = std::chrono::high_resolution_clock::now();
    bool success = opt.go();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = t2 - t1;

    if (success)
        std::cerr << "SUCCESS; running time: " << time.count() << " s; min det J = " << opt.detmin << std::endl;
    else
        std::cerr << "FAIL TO UNTANGLE!" << std::endl;

    for (int v : vert_iter(ref))
        for (int d : range(3))
            ref.points[v][d] = opt.X[3*v+d];

    /*{ // restore scale
        double maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
        for (vec3 &p : ref.points)
            p = (p - vec3(1,1,1)*boxsize/2)/boxsize*maxside + (bbmax+bbmin)/2.;
    }*/

    auto cgal_valid = check_validity_with_cgal(ref, res_filename);
    if(cgal_valid){
        std::cout<<" SUCCESS CONFIRMED WITH CGAL"<<std::endl;
        write_by_extension(res_filename, ref, VolumeAttributes{ { {"selection", opt.lock.ptr} }, { {"det", opt.det.ptr} }, {}, {} });
    }else{
        std::cout<<" FAILED TO OBTAIN CGAL-VALID MESH"<<std::endl;
    }



    /*if (inverted)
        for (vec3 &p : ref.points)
            p.x *= -1;*/



    return !cgal_valid;
}

